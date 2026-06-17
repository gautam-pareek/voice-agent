import multiprocessing
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
import yaml

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# PyTorch index URLs per backend
_TORCH_URLS: dict[str, str] = {
    "cuda": "https://download.pytorch.org/whl/cu126",   # NVIDIA — cu126 works with CUDA 12.x and 13.x
    "rocm": "https://download.pytorch.org/whl/rocm6.2", # AMD ROCm
}


@dataclass(frozen=True)
class HardwareInfo:
    tier: int
    cuda_available: bool    # any GPU hardware found
    torch_ready: bool       # torch can actually run inference on that GPU
    backend: str            # 'cuda' | 'rocm' | 'mps' | 'xpu' | 'cpu'
    vram_gb: float
    cpu_cores: int
    cpu_threads: int        # recommended thread count for faster-whisper CPU inference
    ram_gb: float
    recommended_stt: str
    recommended_tts: str


# ---------------------------------------------------------------------------
# GPU detection — hardware-level, no PyTorch dependency
# ---------------------------------------------------------------------------

def _try_nvidia() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip().split("\n")[0]
        return round(float(out.strip()) / 1024, 2)
    except Exception:
        return None


def _try_amd() -> float | None:
    try:
        import json
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        data = json.loads(out)
        for card in data.values():
            if isinstance(card, dict) and "VRAM Total Memory (B)" in card:
                return round(int(card["VRAM Total Memory (B)"]) / (1024 ** 3), 2)
        return None
    except Exception:
        return None


def _try_apple_silicon() -> float:
    """Unified memory on Apple Silicon — 1/4 of total RAM treated as VRAM equivalent."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return round(int(out) / (1024 ** 3) / 4, 2)
    except Exception:
        return 0.0


def _try_wmi_discrete() -> float | None:
    """Windows fallback: any discrete GPU via WMIC. Skips integrated and basic display adapters."""
    _SKIP = {"microsoft basic display", "intel(r) uhd", "intel(r) hd graphics", "vmware", "virtualbox"}
    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "AdapterRAM,Name", "/format:list"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace")
        entry: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                entry[k.strip()] = v.strip()
            elif not line.strip() and entry:
                name = entry.get("Name", "").lower()
                ram_str = entry.get("AdapterRAM", "0") or "0"
                if not any(s in name for s in _SKIP) and int(ram_str) > 0:
                    return round(int(ram_str) / (1024 ** 3), 2)
                entry = {}
        return None
    except Exception:
        return None


def _query_gpu() -> tuple[bool, float, str]:
    """Return (gpu_found, vram_gb, backend) for any GPU type.

    Tries in order: NVIDIA → AMD → Apple Silicon → Intel XPU → Windows WMI fallback.
    """
    vram = _try_nvidia()
    if vram is not None:
        return True, vram, "cuda"

    vram = _try_amd()
    if vram is not None:
        return True, vram, "rocm"

    if sys.platform == "darwin" and platform.machine() == "arm64":
        return True, _try_apple_silicon(), "mps"

    if _TORCH_AVAILABLE and hasattr(torch, "xpu") and torch.xpu.is_available():
        return True, 0.0, "xpu"

    if sys.platform == "win32":
        vram = _try_wmi_discrete()
        if vram is not None:
            return True, vram, "cuda"

    return False, 0.0, "cpu"


# ---------------------------------------------------------------------------
# PyTorch readiness check per backend
# ---------------------------------------------------------------------------

def _torch_ready(backend: str) -> bool:
    if not _TORCH_AVAILABLE:
        return False
    if backend in ("cuda", "rocm"):
        return torch.cuda.is_available()
    if backend == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if backend == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return True  # cpu — always ready


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AUTO_FIX_ENV = "VOICE_AGENT_AUTO_FIX_DONE"


def ensure_ready() -> None:
    """Call at CLI startup.

    Detects GPU hardware, and if PyTorch can't use it, auto-installs the right
    CUDA/ROCm build then re-execs the current command transparently.
    MPS (Apple Silicon) and XPU (Intel) work out of the box with standard torch.

    The env var VOICE_AGENT_AUTO_FIX_DONE prevents an infinite restart loop —
    if the install failed (e.g. no wheel for this Python version), we warn once
    and continue in CPU mode rather than looping.
    """
    if os.environ.get(_AUTO_FIX_ENV):
        # Already attempted a fix in a previous exec — don't loop.
        if not _torch_ready("cuda") and not _torch_ready("rocm"):
            print(
                "[voice-agent] Warning: GPU detected but CUDA/ROCm PyTorch could not be installed "
                "(your Python version may not have wheels yet). Running on CPU."
            )
        return

    gpu_found, _, backend = _query_gpu()
    if not gpu_found or backend in ("cpu", "mps", "xpu"):
        return
    if _torch_ready(backend):
        return

    url = _TORCH_URLS.get(backend)
    if url is None:
        return

    vendor = {"cuda": "NVIDIA", "rocm": "AMD"}.get(backend, backend.upper())
    print(f"[voice-agent] {vendor} GPU detected but PyTorch has no {backend.upper()} support. Auto-installing (one-time)...")

    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",  # replace +cpu build even if version number matches
        "torch", "--index-url", url,
    ])

    if result.returncode != 0:
        print(f"[voice-agent] Auto-install failed. To enable GPU manually:\n  pip install --force-reinstall torch --index-url {url}")
        return

    # Verify it actually works in a fresh interpreter before restarting.
    # If no wheel exists for this Python version, torch.cuda still returns False here.
    check = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=30,
    )
    if check.stdout.strip() != "True":
        print(
            "[voice-agent] Installed from CUDA index but CUDA is still not available.\n"
            "  This usually means no CUDA wheel exists for your Python version yet.\n"
            f"  Manual fix when wheels are available:\n    pip install --force-reinstall torch --index-url {url}\n"
            "  Running on CPU for now."
        )
        return

    print("[voice-agent] Done. Restarting...")
    new_env = {**os.environ, _AUTO_FIX_ENV: "1"}
    os.execve(sys.executable, [sys.executable, "-m", "voice_agent"] + sys.argv[1:], new_env)


def detect() -> HardwareInfo:
    """Detect hardware tier. Call ensure_ready() first in CLI contexts."""
    gpu_found, vram_gb, backend = _query_gpu()
    ready = _torch_ready(backend) if gpu_found else True

    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    cpu_cores = multiprocessing.cpu_count()
    # Use half the cores for inference threads — leaves headroom for the OS and UI
    cpu_threads = max(1, cpu_cores // 2)
    tiers = _load_tiers()

    if gpu_found and vram_gb >= 8.0:
        tier = 1
    elif gpu_found and vram_gb >= 4.0:
        tier = 2
    elif gpu_found and vram_gb >= 2.0:
        tier = 3
    elif not gpu_found and ram_gb >= 16.0 and cpu_cores >= 8:
        tier = 3  # powerful CPU: enough RAM and cores to run whisper-base
    elif not gpu_found and ram_gb >= 8.0 and cpu_cores >= 4:
        tier = 4  # standard CPU: whisper-tiny
    else:
        tier = 4  # low-end CPU or very little RAM: whisper-tiny

    cfg = tiers[f"tier{tier}"]
    return HardwareInfo(
        tier=tier,
        cuda_available=gpu_found,
        torch_ready=ready,
        backend=backend,
        vram_gb=vram_gb,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        recommended_stt=cfg["stt"],
        recommended_tts=cfg["tts"],
    )


def _load_tiers() -> dict:
    path = Path(__file__).parent.parent / "config" / "tiers.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
