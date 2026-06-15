from voice_agent.core.hardware import HardwareInfo, detect


def test_detect_returns_valid_tier():
    info = detect()
    assert isinstance(info, HardwareInfo)
    assert info.tier in (1, 2, 3, 4)
    assert info.cpu_cores > 0
    assert info.ram_gb > 0
    assert info.recommended_stt in ("large-v3", "medium", "base", "tiny")
    assert info.recommended_tts in ("xtts-v2", "f5-tts", "piper")


def test_detect_is_immutable():
    info = detect()
    try:
        info.tier = 99  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass


def test_cuda_and_vram_consistent():
    info = detect()
    if not info.cuda_available:
        assert info.vram_gb == 0.0
        assert info.tier == 4
