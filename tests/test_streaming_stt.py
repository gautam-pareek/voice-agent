"""Tests for the DIY real-time STT implementation (streaming_stt.py).

Architecture under test:
  - Single tiny whisper model (no RealtimeSTT subprocess)
  - Our own sounddevice + silero-VAD recording loop
  - Three stop triggers: VAD silence, SPACE key, voice command
  - Pure-function helpers: _ends_with_stop_phrase, _strip_stop_phrase
"""
import os
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import voice_agent.core.streaming_stt as m
from voice_agent.core.streaming_stt import (
    StreamingRecorder,
    _ends_with_stop_phrase,
    _strip_stop_phrase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(skip_confidence: bool = False) -> dict:
    return {
        "stt": {
            "realtime_model": "tiny",
            "streaming_skip_confidence_pass": skip_confidence,
            "language": "en",
        }
    }


def _hw_info(torch_ready: bool = False, tier: int = 2) -> MagicMock:
    info = MagicMock()
    info.tier = tier
    info.torch_ready = torch_ready
    return info


class _FakeStream:
    """Fake sounddevice.InputStream: calls the audio callback with silent chunks on enter."""

    def __init__(self, *, samplerate=None, channels=None, dtype=None,
                 blocksize=None, callback=None, **kw):
        self._cb = callback

    def __enter__(self):
        # Inject 5 silent audio chunks so the VAD/transcription threads have data
        for _ in range(5):
            chunk = np.zeros((512, 1), dtype=np.float32)
            self._cb(chunk, 512, None, None)
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# 1–3. _ends_with_stop_phrase — pure function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("hello stop listening", True),
    ("please stop recording", True),
    ("please pause recording", True),  # "pause recording" is a stop phrase
    ("hello world",          False),
    ("",                     False),
    ("stopping the car",     False),  # "stopping" ≠ "stop"
])
def test_ends_with_stop_phrase(text, expected):
    assert _ends_with_stop_phrase(text) == expected


def test_ends_with_stop_phrase_ignores_trailing_punctuation():
    # Whisper sometimes appends a period
    assert _ends_with_stop_phrase("hello stop listening.") is True
    assert _ends_with_stop_phrase("hello stop listening!") is True


# ---------------------------------------------------------------------------
# 4–6. _strip_stop_phrase — pure function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("hello stop listening",  "hello"),
    ("please stop recording", "please"),
    ("hello world",           "hello world"),  # no phrase → unchanged
    ("stop listening",        ""),              # phrase only → empty string
])
def test_strip_stop_phrase(text, expected):
    assert _strip_stop_phrase(text) == expected


# ---------------------------------------------------------------------------
# 7. stop() is a no-op — safe to call multiple times
# ---------------------------------------------------------------------------

def test_stop_is_idempotent():
    rec = StreamingRecorder()
    rec.stop()
    rec.stop()
    rec.stop()  # must not raise


# ---------------------------------------------------------------------------
# 8. Cached model returned on second call without re-loading
# ---------------------------------------------------------------------------

def test_model_cache_returns_same_instance(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(m, "_tiny_model", fake)
    assert m._get_tiny_model() is fake


# ---------------------------------------------------------------------------
# 9. Fresh model: device=cpu when torch_ready=False, uses realtime_model name
# ---------------------------------------------------------------------------

def test_model_loaded_with_correct_args(monkeypatch):
    monkeypatch.setattr(m, "_tiny_model", None)

    fake_model = MagicMock()
    fake_fw = MagicMock()
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_fw}):
        with patch.object(m.settings, "load", return_value=_cfg()):
            with patch("voice_agent.core.streaming_stt.detect", return_value=_hw_info(torch_ready=False)):
                result = m._get_tiny_model()

    fake_fw.WhisperModel.assert_called_once_with(
        "tiny", device="cpu", compute_type="int8", local_files_only=True
    )
    assert result is fake_model


# ---------------------------------------------------------------------------
# 10. Final model cache
# ---------------------------------------------------------------------------

def test_final_model_cached_on_second_call(monkeypatch):
    """_get_final_model() returns the cached instance without reloading."""
    fake = MagicMock()
    monkeypatch.setattr(m, "_final_model", fake)
    assert m._get_final_model() is fake


# ---------------------------------------------------------------------------
# 11. _transcribe_final_accurate joins segment texts
# ---------------------------------------------------------------------------

def test_transcribe_final_accurate_joins_segments(monkeypatch):
    fake_seg1, fake_seg2 = MagicMock(), MagicMock()
    fake_seg1.text, fake_seg2.text = " Hello", " world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([fake_seg1, fake_seg2], MagicMock())
    monkeypatch.setattr(m, "_get_final_model", lambda: mock_model)

    result = m._transcribe_final_accurate(np.zeros(1024, dtype=np.float32))
    assert result == "Hello world"


# ---------------------------------------------------------------------------
# 12. Integration: listen_push_to_talk() completes and returns a valid result.
#
# Stop mechanism: voice command ("hello stop listening" → stop_event set).
# All I/O is mocked: sounddevice, VAD, WhisperModel, _wait_for_key, msvcrt.
# ---------------------------------------------------------------------------

def test_listen_push_to_talk_returns_valid_result(monkeypatch):
    # ------ mock tiny model ------
    fake_seg = MagicMock()
    fake_seg.text = " hello stop listening"
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = lambda audio, **kw: ([fake_seg], MagicMock())
    monkeypatch.setattr(m, "_get_tiny_model", lambda: mock_model)

    # ------ mock VAD: first chunk fires "start", rest return None ------
    mock_vad_model = MagicMock()
    monkeypatch.setattr(m, "_get_vad_model", lambda: mock_vad_model)

    call_count = [0]
    def _fake_vad(chunk, return_seconds=False):
        call_count[0] += 1
        return {"start": 0} if call_count[0] == 1 else None

    mock_vad_iter = MagicMock(side_effect=_fake_vad)

    # ------ mock sounddevice ------
    fake_sd = MagicMock()
    fake_sd.InputStream = _FakeStream

    received_partials: list[str] = []

    with patch("silero_vad.VADIterator", return_value=mock_vad_iter), \
         patch("voice_agent.core.audio._wait_for_key"), \
         patch("voice_agent.core.audio._require_sounddevice", return_value=fake_sd), \
         patch.object(m.settings, "load", return_value=_cfg()), \
         patch("voice_agent.core.streaming_stt._transcribe_final_accurate", return_value="hello"), \
         patch("builtins.print"), \
         patch("msvcrt.kbhit", return_value=False):

        rec = StreamingRecorder(on_partial=received_partials.append)
        result, audio = rec.listen_push_to_talk()

    # Shape checks
    assert isinstance(result["text"], str)
    assert result["text"] == "hello"          # "stop listening" stripped by _strip_stop_phrase
    assert result["language"] == "en"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["segments"], list)
    assert result["audio_path_temp"] is not None
    assert isinstance(audio, np.ndarray)

    # WAV file must exist and be valid
    path = result["audio_path_temp"]
    assert os.path.exists(path)
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2   # 16-bit
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
    os.unlink(path)

    # on_partial callback must have been called at least once
    assert len(received_partials) >= 1
