# rl_loop.py — Correction prompt and RL pipeline coordinator.
#
# Called after every STT transcription. If confidence is below threshold (D004),
# shows a correction prompt. Accepted corrections pass through the novelty gate
# (D006) before being stored. After storage, maybe_trigger() is called so the
# trainer can decide whether to fire a fine-tune run (D007).
#
# Public API:
#   process(result, audio) — the single entry point; called by cli.py and later
#                            by mcp_server.py (Phase 7) and rest_api.py (Phase 8)

import sys
from typing import TYPE_CHECKING

import numpy as np

from voice_agent.config import settings

if TYPE_CHECKING:
    from voice_agent.core.stt import TranscriptResult


def process(result: "TranscriptResult", audio: np.ndarray) -> None:
    """Decide whether to prompt for correction, then store and trigger if needed.

    Exits silently when:
    - confidence >= threshold (D004 — model is confident, no prompt)
    - stdin is not a tty (non-interactive context: MCP/REST/pipe)
    - user accepts the transcription as-is
    - the pair is not novel (D006 — model already knows this pattern)
    """
    cfg = settings.load()
    threshold: float = cfg["stt"]["confidence_threshold"]

    if result["confidence"] >= threshold:
        return

    if not _is_interactive():
        return

    predicted = result["text"]
    correction = _prompt(predicted, result["confidence"])

    if correction is None:
        return  # user accepted — nothing to store

    # Import here to avoid circular top-level imports
    from voice_agent.learning import novelty, store, trainer

    if not novelty.is_novel(audio, predicted):
        _print_err("(skipped — model already knows this pattern)")
        return

    audio_path = result.get("audio_path_temp")
    if audio_path is None:
        _print_err("(skipped — no audio file available for storage)")
        return

    pair_id = store.save_pair(
        audio_path=audio_path,
        predicted=predicted,
        corrected=correction,
        confidence=result["confidence"],
    )
    _print_err(f"Correction #{pair_id} saved  ({store.count_pending()} pending)")

    trainer.maybe_trigger()


# --- internal helpers ---

def _prompt(predicted: str, confidence: float) -> str | None:
    """Show the correction prompt; return correction string or None (accepted)."""
    _print_err(f"\n[low confidence: {confidence:.2f}]")
    _print_err(f"Heard: '{predicted}'")
    sys.stderr.write("Press Enter to accept, or type correction: ")
    sys.stderr.flush()

    raw = sys.stdin.readline()
    correction = raw.strip()

    if not correction:
        return None  # accepted as-is

    if correction == predicted:
        return None  # typed the same text — treat as acceptance

    return correction


def _is_interactive() -> bool:
    """Return True when both stdin and stderr are connected to a real terminal."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)
