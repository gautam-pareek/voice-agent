# stt.py — faster-whisper wrapper, tier-aware model loading.
#
# Public API:
#   TranscriptResult  — TypedDict with text, language, confidence, segments, audio_path_temp
#   transcribe()      — transcribe a float32 16 kHz mono numpy array, return TranscriptResult

import math
from typing import Callable, TypedDict

import numpy as np

from voice_agent.config import settings
from voice_agent.core.hardware import detect

_model_cache: dict[str, object] = {}


class TranscriptResult(TypedDict):
    text: str
    language: str
    confidence: float          # 0.0–1.0, exp(avg_log_prob) across all segments
    segments: list[dict]       # [{start, end, text, avg_log_prob}, ...]
    audio_path_temp: str | None  # path to WAV file written by audio.py (Phase 3 uses this)


def transcribe(
    audio: np.ndarray,
    audio_path: str | None = None,
    on_segment: Callable[[str], None] | None = None,
) -> TranscriptResult:
    """Transcribe a float32 16 kHz mono audio array.

    Args:
        audio       — numpy float32 array, 16 kHz, mono
        audio_path  — optional path to the WAV file already on disk (forwarded to result)
        on_segment  — optional callback called with each segment's text as it is decoded;
                      use this to print streaming output in the CLI

    Returns:
        TranscriptResult with text, language, confidence (0–1), segments, audio_path_temp.
    """
    model = _load_model()
    cfg = settings.load()
    language: str = cfg["stt"]["language"]

    segments_iter, info = model.transcribe(audio, language=language)

    seg_dicts: list[dict] = []
    text_parts: list[str] = []
    log_prob_sum = 0.0

    for seg in segments_iter:
        seg_dicts.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_log_prob": seg.avg_log_prob,
            }
        )
        text_parts.append(seg.text)
        log_prob_sum += seg.avg_log_prob

        if on_segment is not None:
            on_segment(seg.text)

    text = "".join(text_parts).strip()
    seg_count = max(len(seg_dicts), 1)
    # exp(avg_log_prob) converts a negative log-prob to a 0–1 confidence value.
    # Typical good transcriptions have avg_log_prob around -0.1 to -0.3 → confidence 0.74–0.90.
    confidence = float(math.exp(log_prob_sum / seg_count))

    return TranscriptResult(
        text=text,
        language=info.language,
        confidence=confidence,
        segments=seg_dicts,
        audio_path_temp=audio_path,
    )


def _load_model():
    """Load (or retrieve from cache) the WhisperModel for the current config/tier.

    faster-whisper downloads the checkpoint automatically on first use.
    The loaded model is cached at module level to avoid reloading on every call.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )

    model_name, device, compute_type = _resolve_model_params()
    cache_key = f"{model_name}:{device}"

    if cache_key not in _model_cache:
        _model_cache[cache_key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )

    return _model_cache[cache_key]


def _resolve_model_params() -> tuple[str, str, str]:
    """Return (model_name, device, compute_type) from config or tier defaults.

    If config["stt"]["model"] is None, the tier's recommended model is used.
    compute_type is int8_float16 on CUDA (fast + low VRAM), int8 on CPU.
    """
    cfg = settings.load()
    model_name: str | None = cfg["stt"]["model"]

    info = detect()
    if model_name is None:
        model_name = info.recommended_stt

    device = "cuda" if info.cuda_available else "cpu"
    compute_type = "int8_float16" if info.cuda_available else "int8"

    return model_name, device, compute_type
