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

# Words the user can type to explicitly discard a sample (don't train on it).
_DISCARD_WORDS = frozenset({
    "s", "skip", "discard", "null", "none", "n", "d",
    "delete", "nothing", "empty", "no", "nope",
})


def process(result: "TranscriptResult", audio: np.ndarray) -> None:
    """Decide whether to prompt for correction, then store and trigger if needed.

    Exits silently when:
    - text is empty (no speech detected)
    - confidence >= threshold (D004 — model is confident, no prompt)
    - stdin is not a tty (non-interactive context: MCP/REST/pipe)
    - user accepts the transcription as-is or discards it
    - the pair is not novel (D006 — model already knows this pattern)
    """
    cfg = settings.load()
    threshold: float = cfg["stt"]["confidence_threshold"]

    # Skip if nothing was heard
    if not result["text"].strip():
        return

    if result["confidence"] >= threshold:
        return

    if not _is_interactive():
        return

    predicted = result["text"]
    correction, should_discard = _prompt(predicted, result["confidence"])

    if should_discard:
        _print_err("(sample discarded — not saved for training)")
        return

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

def _prompt(predicted: str, confidence: float) -> tuple[str | None, bool]:
    """Show the correction prompt.

    Returns (correction, should_discard):
      correction=None, should_discard=False  → user accepted as-is
      correction=None, should_discard=True   → user explicitly discarded
      correction=str,  should_discard=False  → user provided a correction
    """
    _print_err(f"\n[low confidence: {confidence:.2f}]")
    _print_err(f"  {predicted}")
    _print_err("")
    _print_err("  Enter         → accept as-is")
    _print_err("  s + Enter     → skip (bad audio / silence / noise)")
    _print_err("  your text...  → type the correction and press Enter")

    raw = _editable_input(predicted)
    correction = raw.strip()

    if correction.lower() in _DISCARD_WORDS:
        return None, True  # discard

    if not correction or correction == predicted:
        return None, False  # accepted as-is

    return correction, False


def _editable_input(prefill: str) -> str:
    """Input line pre-filled with `prefill` so the user can edit rather than retype.

    Uses readline when available (Linux/macOS). Falls back to plain input on
    Windows (user can still backspace and retype — just no arrow-key navigation).
    """
    try:
        import readline as rl
        rl.set_startup_hook(lambda: rl.insert_text(prefill))
        try:
            return input("> ")
        finally:
            rl.set_startup_hook()
    except (ImportError, AttributeError):
        sys.stderr.write("> ")
        sys.stderr.flush()
        return sys.stdin.readline()


def _is_interactive() -> bool:
    """Return True when both stdin and stderr are connected to a real terminal."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)
