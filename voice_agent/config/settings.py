# settings.py — Config loader and writer for voice-agent.
#
# Every module in this project calls load() to get user settings instead of
# hardcoding values. This keeps all configuration in one place: ~/.voice_agent/config.json.
#
# Public API:
#   DATA_DIR    — Path to ~/.voice_agent/ (all user data lives here)
#   CONFIG_PATH — Path to ~/.voice_agent/config.json
#   load()      — Returns merged dict of defaults + whatever the user has saved
#   save(cfg)   — Writes cfg to CONFIG_PATH (creates the directory if missing)

import json
from pathlib import Path
from typing import Any

# ~/.voice_agent/ — all persistent user data (config, voice profile, adapters)
DATA_DIR = Path.home() / ".voice_agent"

# The single config file for all settings
CONFIG_PATH = DATA_DIR / "config.json"

# Defaults for every setting the app understands.
# null means "auto-detect at runtime" (e.g. tier is set by hardware.py on first run).
# These are never written to disk — they only fill gaps when the user's config is missing keys.
_DEFAULTS: dict[str, Any] = {
    "tier": None,                        # hardware tier 1-4, auto-detected if null
    "stt": {
        "model": None,                   # whisper checkpoint, set by tier if null
        "language": "en",               # locked to English in V1
        "confidence_threshold": 0.75,   # RL correction prompt fires below this score
        "realtime_model": "tiny",        # whisper model for streaming partials (D020)
        "partial_update_ms": 300,        # how often partial transcripts refresh during speech
        "streaming_skip_confidence_pass": False,  # if True, skip RL prompt after streaming finals
    },
    "tts": {
        "engine": None,                  # "xtts", "f5", or "piper" — set by tier if null
        "voice_pack": "en-ryan-medium", # default Piper voice pack for tier 3/4
    },
    "rl": {
        "min_samples_to_train": 50,      # fine-tune runs after this many corrections (D007)
        "preferred_batch_size": 50,      # soft target before triggering a scheduled retrain
        "novelty_speaker_threshold": 0.85,   # cosine similarity above this = not novel (D006)
        "novelty_semantic_threshold": 0.90,  # cosine similarity above this = not novel (D006)
    },
    "api": {
        "rest_port": 7532,               # FastAPI REST server port (Phase 8)
    },
}


def load() -> dict[str, Any]:
    """Return the active config: defaults deep-merged with whatever is stored on disk.

    If no config file exists yet (first run), returns pure defaults.
    Stored values always win over defaults for matching keys.
    Nested dicts are merged key-by-key, so the user can override one stt field
    without wiping out the others.
    """
    if not CONFIG_PATH.exists():
        # First run — write defaults to disk so the user can inspect and edit them
        defaults = _merge(_DEFAULTS, {})
        save(defaults)
        return defaults

    with open(CONFIG_PATH) as f:
        stored = json.load(f)

    # Merge so that any keys the user hasn't set fall back to _DEFAULTS
    return _merge(_DEFAULTS, stored)


def save(cfg: dict[str, Any]) -> None:
    """Write cfg to CONFIG_PATH as formatted JSON.

    Creates ~/.voice_agent/ if it doesn't exist yet (e.g. very first save).
    Overwrites the existing file completely — always pass the full config dict.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into defaults and return a new dict.

    Shallow merge would wipe nested dicts (e.g. saving {"stt": {"language": "hi"}}
    would delete confidence_threshold). Deep merge only touches the keys present
    in overrides, leaving all other defaults intact.

    Neither input dict is modified — a new dict is always returned (immutability).
    """
    # Start with a shallow copy of defaults so we don't mutate the original
    result = {**defaults}

    for key, val in overrides.items():
        # If both sides have a dict for this key, recurse instead of overwriting
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge(result[key], val)
        else:
            # Scalar value or a key not in defaults — take the override as-is
            result[key] = val

    return result
