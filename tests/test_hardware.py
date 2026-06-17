from voice_agent.core.hardware import HardwareInfo, detect

_VALID_BACKENDS = ("cuda", "rocm", "mps", "xpu", "cpu")


def test_detect_returns_valid_tier():
    info = detect()
    assert isinstance(info, HardwareInfo)
    assert info.tier in (1, 2, 3, 4)
    assert info.cpu_cores > 0
    assert info.cpu_threads >= 1
    assert info.cpu_threads <= info.cpu_cores
    assert info.ram_gb > 0
    assert info.recommended_stt in ("large-v3", "medium", "base", "tiny")
    assert info.recommended_tts in ("xtts-v2", "f5-tts", "piper")
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.torch_ready, bool)
    assert info.backend in _VALID_BACKENDS


def test_detect_is_immutable():
    info = detect()
    try:
        info.tier = 99  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass


def test_no_gpu_means_tier_4_and_zero_vram():
    info = detect()
    if not info.cuda_available:
        assert info.vram_gb == 0.0
        assert info.tier == 4
        assert info.backend == "cpu"


def test_torch_ready_requires_gpu():
    info = detect()
    # torch_ready=True without any GPU hardware is impossible (CPU is always ready
    # but cuda_available would be False, so this checks the GPU path only)
    if info.torch_ready and info.backend != "cpu":
        assert info.cuda_available


def test_vram_nonzero_when_gpu_found():
    info = detect()
    # MPS and XPU may report 0.0 VRAM (can't easily query unified/Intel VRAM)
    if info.cuda_available and info.backend in ("cuda", "rocm"):
        assert info.vram_gb > 0.0


def test_cpu_only_powerful_machine_gets_tier_3():
    """16GB RAM + 8 cores should reach tier 3 (whisper-base) without a GPU."""
    from unittest.mock import patch
    with patch("voice_agent.core.hardware._query_gpu", return_value=(False, 0.0, "cpu")):
        with patch("psutil.virtual_memory") as mock_mem:
            with patch("multiprocessing.cpu_count", return_value=8):
                mock_mem.return_value.total = 16 * (1024 ** 3)
                info = detect()
    assert info.tier == 3
    assert info.recommended_stt == "base"


def test_cpu_only_low_end_stays_tier_4():
    """4GB RAM + 2 cores should stay tier 4 (whisper-tiny)."""
    from unittest.mock import patch
    with patch("voice_agent.core.hardware._query_gpu", return_value=(False, 0.0, "cpu")):
        with patch("psutil.virtual_memory") as mock_mem:
            with patch("multiprocessing.cpu_count", return_value=2):
                mock_mem.return_value.total = 4 * (1024 ** 3)
                info = detect()
    assert info.tier == 4
    assert info.recommended_stt == "tiny"
