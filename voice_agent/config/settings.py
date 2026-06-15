import json
from pathlib import Path
from typing import Any

DATA_DIR = Path.home() / ".voice_agent"
CONFIG_PATH = DATA_DIR / "config.json"

_DEFAULTS: dict[str, Any] = {
    "tier": None,
    "stt": {
        "model": None,
        "language": "en",
        "confidence_threshold": 0.75,
    },
    "tts": {
        "engine": None,
        "voice_pack": "en-ryan-medium",
    },
    "rl": {
        "min_samples_to_train": 1,
        "preferred_batch_size": 50,
        "novelty_speaker_threshold": 0.85,
        "novelty_semantic_threshold": 0.90,
    },
    "api": {
        "rest_port": 7532,
    },
}


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return _merge(_DEFAULTS, {})
    with open(CONFIG_PATH) as f:
        stored = json.load(f)
    return _merge(_DEFAULTS, stored)


def save(cfg: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = {**defaults}
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge(result[key], val)
        else:
            result[key] = val
    return result
