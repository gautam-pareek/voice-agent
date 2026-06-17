import json
import wave
import tempfile
from pathlib import Path

import numpy as np
import pytest

from voice_agent.learning.store import (
    PendingPair,
    clear_pending,
    count_pending,
    load_corpus_embeddings,
    load_pending,
    save_corpus_embeddings,
    save_pair,
    PENDING_DIR,
    EMBED_DIR,
    ADAPTER_DIR,
)


# --- fixtures ---

@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Redirect all store paths to a temp directory for each test."""
    import voice_agent.learning.store as store_mod
    import voice_agent.config.settings as settings_mod

    monkeypatch.setattr(store_mod, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(store_mod, "EMBED_DIR",   tmp_path / "embeddings")
    monkeypatch.setattr(store_mod, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(store_mod, "_CORPUS_PATH", tmp_path / "embeddings" / "corpus.npz")
    monkeypatch.setattr(settings_mod, "DATA_DIR",  tmp_path)


def _make_wav(path: str) -> None:
    """Write a minimal silent WAV to path."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 3200)


# --- ensure_dirs ---

def test_ensure_dirs_creates_all_directories():
    from voice_agent.learning.store import ensure_dirs, PENDING_DIR, EMBED_DIR, ADAPTER_DIR
    ensure_dirs()
    assert PENDING_DIR.exists()
    assert EMBED_DIR.exists()
    assert ADAPTER_DIR.exists()


# --- save_pair / count_pending ---

def test_save_pair_creates_wav_and_meta(tmp_path):
    from voice_agent.learning.store import ensure_dirs, PENDING_DIR
    ensure_dirs()

    src = tmp_path / "audio.wav"
    _make_wav(str(src))

    pair_id = save_pair(str(src), predicted="hello word", corrected="hello world", confidence=0.61)

    assert pair_id == 1
    assert (PENDING_DIR / "0001_audio.wav").exists()
    meta_path = PENDING_DIR / "0001_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["predicted"] == "hello word"
    assert meta["corrected"] == "hello world"
    assert meta["confidence"] == 0.61


def test_count_pending_increments():
    from voice_agent.learning.store import ensure_dirs
    import tempfile
    ensure_dirs()

    assert count_pending() == 0

    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _make_wav(f.name)
            save_pair(f.name, predicted=f"pred{i}", corrected=f"corr{i}", confidence=0.5)

    assert count_pending() == 3


def test_save_pair_ids_are_sequential():
    from voice_agent.learning.store import ensure_dirs
    import tempfile
    ensure_dirs()

    ids = []
    for _ in range(3):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _make_wav(f.name)
            ids.append(save_pair(f.name, predicted="a", corrected="b", confidence=0.5))

    assert ids == [1, 2, 3]


# --- load_pending ---

def test_load_pending_returns_sorted_pairs():
    from voice_agent.learning.store import ensure_dirs
    import tempfile
    ensure_dirs()

    for word in ("apple", "banana", "cherry"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _make_wav(f.name)
            save_pair(f.name, predicted=word, corrected=word + "x", confidence=0.5)

    pairs = load_pending()
    assert len(pairs) == 3
    assert pairs[0]["predicted"] == "apple"
    assert pairs[1]["predicted"] == "banana"
    assert pairs[2]["predicted"] == "cherry"


def test_load_pending_returns_empty_when_no_pairs():
    pairs = load_pending()
    assert pairs == []


# --- clear_pending ---

def test_clear_pending_deletes_wav_and_json():
    from voice_agent.learning.store import ensure_dirs, PENDING_DIR
    import tempfile
    ensure_dirs()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        save_pair(f.name, predicted="test", corrected="best", confidence=0.5)

    assert count_pending() == 1
    clear_pending()
    assert count_pending() == 0
    assert list(PENDING_DIR.glob("*.wav")) == []
    assert list(PENDING_DIR.glob("*.json")) == []


# --- corpus embeddings ---

def test_corpus_embeddings_round_trip():
    from voice_agent.learning.store import ensure_dirs
    ensure_dirs()

    arr = np.random.rand(5, 66).astype(np.float32)
    save_corpus_embeddings(arr)

    loaded = load_corpus_embeddings()
    assert loaded is not None
    assert loaded.shape == (5, 66)
    np.testing.assert_allclose(loaded, arr)


def test_load_corpus_embeddings_returns_none_when_absent():
    result = load_corpus_embeddings()
    assert result is None
