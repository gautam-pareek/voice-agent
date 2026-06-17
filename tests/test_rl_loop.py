import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_agent.core.stt import TranscriptResult
from voice_agent.learning import rl_loop


def _make_result(text: str = "hello world", confidence: float = 0.5, audio_path: str = "/tmp/a.wav") -> TranscriptResult:
    return TranscriptResult(
        text=text,
        language="en",
        confidence=confidence,
        segments=[],
        audio_path_temp=audio_path,
    )


def _cfg(threshold: float = 0.75) -> dict:
    return {"stt": {"confidence_threshold": threshold}}


# --- high confidence: no prompt ---

def test_process_skips_when_confidence_above_threshold():
    result = _make_result(confidence=0.9)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt") as mock_prompt:
                rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_prompt.assert_not_called()


# --- non-interactive: no prompt ---

def test_process_skips_when_not_interactive():
    result = _make_result(confidence=0.5)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=False):
            with patch("voice_agent.learning.rl_loop._prompt") as mock_prompt:
                rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_prompt.assert_not_called()


# --- empty text: no prompt (silence / no speech) ---

def test_process_skips_when_text_is_empty():
    result = _make_result(text="", confidence=0.0)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt") as mock_prompt:
                rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_prompt.assert_not_called()


# --- user accepts (no correction): nothing stored ---

def test_process_stores_nothing_on_acceptance():
    result = _make_result(confidence=0.5)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt", return_value=(None, False)):
                with patch("voice_agent.learning.novelty.is_novel") as mock_novel:
                    rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_novel.assert_not_called()


# --- user discards: nothing stored ---

def test_process_stores_nothing_on_discard():
    result = _make_result(confidence=0.5)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt", return_value=(None, True)):
                with patch("voice_agent.learning.novelty.is_novel") as mock_novel:
                    rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_novel.assert_not_called()


# --- not novel: nothing stored ---

def test_process_stores_nothing_when_not_novel():
    result = _make_result(confidence=0.5)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt", return_value=("hello world", False)):
                with patch("voice_agent.learning.novelty.is_novel", return_value=False):
                    with patch("voice_agent.learning.store.save_pair") as mock_save:
                        rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_save.assert_not_called()


# --- novel correction: stored + maybe_trigger called ---

def test_process_stores_pair_and_triggers_when_novel():
    result = _make_result(confidence=0.5, audio_path="/tmp/test.wav")
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt", return_value=("hello world", False)):
                with patch("voice_agent.learning.novelty.is_novel", return_value=True):
                    with patch("voice_agent.learning.store.save_pair", return_value=1) as mock_save:
                        with patch("voice_agent.learning.store.count_pending", return_value=1):
                            with patch("voice_agent.learning.trainer.maybe_trigger") as mock_trigger:
                                rl_loop.process(result, np.zeros(16000, dtype=np.float32))

    mock_save.assert_called_once_with(
        audio_path="/tmp/test.wav",
        predicted="hello world",
        corrected="hello world",
        confidence=0.5,
    )
    mock_trigger.assert_called_once()


# --- no audio path: nothing stored ---

def test_process_skips_storage_when_no_audio_path():
    result = _make_result(confidence=0.5, audio_path=None)
    with patch("voice_agent.learning.rl_loop.settings.load", return_value=_cfg(0.75)):
        with patch("voice_agent.learning.rl_loop._is_interactive", return_value=True):
            with patch("voice_agent.learning.rl_loop._prompt", return_value=("correction", False)):
                with patch("voice_agent.learning.novelty.is_novel", return_value=True):
                    with patch("voice_agent.learning.store.save_pair") as mock_save:
                        rl_loop.process(result, np.zeros(16000, dtype=np.float32))
    mock_save.assert_not_called()


# --- _prompt: enter key → (None, False) ---

def test_prompt_returns_none_on_empty_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    correction, discard = rl_loop._prompt("hello word", 0.5)
    assert correction is None
    assert discard is False


# --- _prompt: correction text → (text, False) ---

def test_prompt_returns_correction(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("hello world\n"))
    correction, discard = rl_loop._prompt("hello word", 0.5)
    assert correction == "hello world"
    assert discard is False


# --- _prompt: same text as predicted → (None, False) ---

def test_prompt_treats_identical_input_as_acceptance(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("hello word\n"))
    correction, discard = rl_loop._prompt("hello word", 0.5)
    assert correction is None
    assert discard is False


# --- _prompt: discard word → (None, True) ---

def test_prompt_returns_discard_on_skip_word(monkeypatch):
    for word in ("s", "skip", "null", "discard", "n"):
        monkeypatch.setattr("sys.stdin", io.StringIO(f"{word}\n"))
        correction, discard = rl_loop._prompt("hello word", 0.5)
        assert correction is None
        assert discard is True, f"Expected discard=True for input '{word}'"
