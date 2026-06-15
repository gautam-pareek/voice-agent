import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import psutil
import yaml

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class HardwareInfo:
    tier: int
    cuda_available: bool
    vram_gb: float
    cpu_cores: int
    ram_gb: float
    recommended_stt: str
    recommended_tts: str


def detect() -> HardwareInfo:
    cuda = _TORCH_AVAILABLE and torch.cuda.is_available()
    vram_gb = _vram_gb() if cuda else 0.0
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    cpu_cores = multiprocessing.cpu_count()

    tiers = _load_tiers()

    if cuda and vram_gb >= 8.0:
        tier = 1
    elif cuda and vram_gb >= 4.0:
        tier = 2
    elif cuda and vram_gb >= 2.0:
        tier = 3
    else:
        tier = 4

    cfg = tiers[f"tier{tier}"]
    return HardwareInfo(
        tier=tier,
        cuda_available=cuda,
        vram_gb=round(vram_gb, 2),
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        recommended_stt=cfg["stt"],
        recommended_tts=cfg["tts"],
    )


def _vram_gb() -> float:
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def _load_tiers() -> dict:
    path = Path(__file__).parent.parent / "config" / "tiers.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
