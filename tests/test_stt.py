from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_agent.core.stt import TranscriptResult, _resolve_model_params, transcribe


# --- TranscriptResult structure ---

def test_transcript_result_has_all_keys():
    result = TranscriptResult(
        text="hello world",
        language="en",
        confidence=0.85,
        segments=[],
        audio_path_temp=None,
    )
    assert result["text"] == "hello world"
    assert result["language"] == "en"
    assert result["confidence"] == 0.85
    assert result["segments"] == []
    assert result["audio_path_temp"] is None


# --- Model param resolution ---

def _mock_no_cuda_detect():
    info = MagicMock()
    info.cuda_available = False
    info.recommended_stt = "base"
    return info


def _mock_cuda_detect():
    info = MagicMock()
    info.cuda_available = True
    info.recommended_stt = "large-v3"
    return info


def test_resolve_model_params_uses_tier_default_when_config_is_none():
    with patch("voice_agent.core.stt.detect", return_value=_mock_no_cuda_detect()):
        with patch(
            "voice_agent.core.stt.settings.load",
            return_value={"stt": {"model": None, "language": "en"}},
        ):
            model_name, device, compute_type = _resolve_model_params()

    assert model_name == "base"
    assert device == "cpu"
    assert compute_type == "int8"


def test_resolve_model_params_respects_config_override():
    with patch("voice_agent.core.stt.detect", return_value=_mock_no_cuda_detect()):
        with patch(
            "voice_agent.core.stt.settings.load",
            return_value={"stt": {"model": "large-v3", "language": "en"}},
        ):
            model_name, _device, _ct = _resolve_model_params()

    assert model_name == "large-v3"


def test_resolve_model_params_uses_cuda_when_available():
    with patch("voice_agent.core.stt.detect", return_value=_mock_cuda_detect()):
        with patch(
            "voice_agent.core.stt.settings.load",
            return_value={"stt": {"model": None, "language": "en"}},
        ):
            _name, device, compute_type = _resolve_model_params()

    assert device == "cuda"
    assert compute_type == "int8_float16"


# --- transcribe() behaviour ---

def _make_mock_segment(text: str, avg_log_prob: float = -0.2) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = 0.0
    seg.end = 1.0
    seg.avg_log_prob = avg_log_prob
    return seg


def _patched_transcribe(segments, language="en"):
    """Context manager helper: patches _load_model and settings.load."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        mock_info = MagicMock()
        mock_info.language = language

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        with patch("voice_agent.core.stt._load_model", return_value=mock_model):
            with patch(
                "voice_agent.core.stt.settings.load",
                return_value={"stt": {"language": "en"}},
            ):
                yield

    return _ctx()


def test_transcribe_returns_correct_text():
    segs = [_make_mock_segment(" hello"), _make_mock_segment(" world")]
    with _patched_transcribe(segs):
        result = transcribe(np.zeros(16000, dtype=np.float32))

    assert result["text"] == "hello world"


def test_transcribe_confidence_in_valid_range():
    segs = [_make_mock_segment(" test", avg_log_prob=-0.3)]
    with _patched_transcribe(segs):
        result = transcribe(np.zeros(16000, dtype=np.float32))

    assert 0.0 < result["confidence"] <= 1.0


def test_transcribe_calls_on_segment_for_each_segment():
    segs = [_make_mock_segment(" one"), _make_mock_segment(" two")]
    collected: list[str] = []

    with _patched_transcribe(segs):
        transcribe(np.zeros(16000, dtype=np.float32), on_segment=collected.append)

    assert collected == [" one", " two"]


def test_transcribe_forwards_audio_path_temp():
    segs = [_make_mock_segment(" hi")]
    with _patched_transcribe(segs):
        result = transcribe(
            np.zeros(16000, dtype=np.float32), audio_path="/tmp/test.wav"
        )

    assert result["audio_path_temp"] == "/tmp/test.wav"


def test_transcribe_empty_audio_returns_empty_text():
    with _patched_transcribe([]):
        result = transcribe(np.zeros(16000, dtype=np.float32))

    assert result["text"] == ""
    assert result["segments"] == []
