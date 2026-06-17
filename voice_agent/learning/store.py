# store.py — Manages ~/.voice_agent/ directory structure and file I/O.
#
# Pure file operations — no AI models, no inference.
#
# Public API:
#   PENDING_DIR, EMBED_DIR, ADAPTER_DIR  — Path constants
#   ensure_dirs()                        — create directory tree on first run
#   save_pair(audio_path, predicted, corrected, confidence) -> int
#   count_pending() -> int
#   load_pending() -> list[PendingPair]
#   clear_pending() -> None              — deletes WAV + JSON (D005)
#   load_corpus_embeddings() -> np.ndarray | None
#   save_corpus_embeddings(embeddings)   — overwrites corpus.npz

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

from voice_agent.config.settings import DATA_DIR

PENDING_DIR = DATA_DIR / "pending"
EMBED_DIR   = DATA_DIR / "embeddings"
ADAPTER_DIR = DATA_DIR / "adapters"

_CORPUS_PATH = EMBED_DIR / "corpus.npz"


class PendingPair(TypedDict):
    pair_id: int
    audio_path: str      # absolute path to WAV inside PENDING_DIR
    predicted: str       # what the model transcribed
    corrected: str       # what the user said it should be
    confidence: float    # stt confidence score at time of capture
    timestamp: str       # ISO-8601 UTC


def ensure_dirs() -> None:
    """Create the full ~/.voice_agent/ tree if any directory is missing."""
    for d in (DATA_DIR, PENDING_DIR, EMBED_DIR, ADAPTER_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_pair(
    audio_path: str,
    predicted: str,
    corrected: str,
    confidence: float,
) -> int:
    """Copy audio into PENDING_DIR and write a metadata JSON alongside it.

    Returns the assigned pair_id (1-indexed, monotonically increasing).
    """
    ensure_dirs()

    pair_id = _next_pair_id()
    dest_audio = PENDING_DIR / f"{pair_id:04d}_audio.wav"
    dest_meta  = PENDING_DIR / f"{pair_id:04d}_meta.json"

    shutil.copy2(audio_path, dest_audio)

    meta: PendingPair = {
        "pair_id":    pair_id,
        "audio_path": str(dest_audio),
        "predicted":  predicted,
        "corrected":  corrected,
        "confidence": round(confidence, 4),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    dest_meta.write_text(json.dumps(meta, indent=2))

    return pair_id


def count_pending() -> int:
    """Return the number of correction pairs waiting in pending/."""
    if not PENDING_DIR.exists():
        return 0
    return len(list(PENDING_DIR.glob("*_meta.json")))


def load_pending() -> list[PendingPair]:
    """Load all pending correction pairs, sorted by pair_id."""
    if not PENDING_DIR.exists():
        return []

    pairs: list[PendingPair] = []
    for meta_file in sorted(PENDING_DIR.glob("*_meta.json")):
        data = json.loads(meta_file.read_text())
        pairs.append(PendingPair(**data))  # type: ignore[misc]

    return pairs


def clear_pending() -> None:
    """Delete all WAV and JSON files in pending/ after a fine-tune cycle.

    Per D005: raw audio is discarded after training; the LoRA adapter
    weights (~2MB) are the permanent artefact.
    """
    if not PENDING_DIR.exists():
        return

    for f in PENDING_DIR.iterdir():
        if f.suffix in (".wav", ".json"):
            f.unlink(missing_ok=True)


def load_corpus_embeddings() -> np.ndarray | None:
    """Return the accumulated embedding matrix (n_pairs × embed_dim), or None.

    Each row is the combined [speaker_emb | semantic_emb] for one stored pair.
    """
    if not _CORPUS_PATH.exists():
        return None

    data = np.load(_CORPUS_PATH)
    return data["embeddings"]


def save_corpus_embeddings(embeddings: np.ndarray) -> None:
    """Overwrite corpus.npz with the updated embedding matrix."""
    ensure_dirs()
    np.savez(_CORPUS_PATH, embeddings=embeddings)


# --- internal helpers ---

def _next_pair_id() -> int:
    """Return 1 + number of existing pairs (so IDs start at 1 and never collide)."""
    return count_pending() + 1
