# streaming_stt.py — DIY real-time STT without RealtimeSTT.
#
# Architecture (D022, D023):
#   sounddevice InputStream → audio_queue → VAD/accumulation thread → transcription thread
#   Main thread polls SPACE key (msvcrt.kbhit) and stop_event.
#
# Models (two, sequential — never both loaded at the same time at peak):
#   _tiny_model  — whisper-tiny (~150 MB VRAM) for real-time display every 300 ms
#   _final_model — configured full model (e.g. whisper-medium, ~2 GB VRAM) for accurate final
#   Both are lazy-loaded singletons freed on process exit.
#   Peak VRAM during finalization: ~2.15 GB (Tier 2). Between calls: ~150 MB.
#
# Stop triggers (first wins):
#   1. silero-VAD detects 1.2 s of silence after speech
#   2. SPACE key pressed  (msvcrt.kbhit in main thread — no subprocess conflicts)
#   3. Transcription ends with a stop phrase ("stop listening", "stop recording", ...)
#
# Public API:
#   StreamingRecorder(on_partial)  — records + shows words in real-time, accurate final
#   StreamingRecorder.listen_push_to_talk() → (TranscriptResult, audio_array)
#   StreamingRecorder.listen()     — alias

import sys
import threading
import time
from queue import Empty, Queue
from typing import Callable

import numpy as np

from voice_agent.config import settings
from voice_agent.core.audio import SAMPLE_RATE, _VAD_CHUNK, _get_vad_model, write_wav
from voice_agent.core.hardware import detect
from voice_agent.core.stt import TranscriptResult

# ---------------------------------------------------------------------------
# Tiny model — real-time display partials (~150 MB VRAM, kept for session)
# ---------------------------------------------------------------------------

_tiny_model = None  # WhisperModel | None


def _get_tiny_model():
    global _tiny_model
    if _tiny_model is None:
        from faster_whisper import WhisperModel

        info = detect()
        device = "cuda" if info.torch_ready else "cpu"
        compute_type = "int8_float16" if info.torch_ready else "int8"
        cfg = settings.load()
        model_name = cfg["stt"].get("realtime_model", "tiny")
        print(f"Loading {model_name} model for real-time display...", flush=True)
        _tiny_model = WhisperModel(
            model_name, device=device, compute_type=compute_type, local_files_only=True
        )
    return _tiny_model


# ---------------------------------------------------------------------------
# Final model — accurate transcript after speech ends (kept for session)
# ---------------------------------------------------------------------------

_final_model = None  # WhisperModel | None


def _get_final_model():
    """Load (or return cached) the configured full STT model for accurate finals.

    Uses the same model name and device as batch-mode stt.py (_resolve_model_params),
    so the tier-recommended model (e.g. whisper-medium on Tier 2) is reused automatically.
    """
    global _final_model
    if _final_model is None:
        from faster_whisper import WhisperModel
        from voice_agent.core.stt import _resolve_model_params

        model_name, device, compute_type = _resolve_model_params()
        print(f"\nLoading {model_name} for accurate final...", end="", flush=True)
        _final_model = WhisperModel(
            model_name, device=device, compute_type=compute_type, local_files_only=True
        )
        print(" ready.", flush=True)
    return _final_model


def _transcribe_final_accurate(audio: np.ndarray) -> str:
    """Re-transcribe the full recorded audio with the accurate full model.

    Returns an empty string if no speech is detected; caller falls back to partial text.
    """
    model = _get_final_model()
    segs, _ = model.transcribe(audio, language="en", beam_size=5, vad_filter=False)
    return "".join(s.text for s in segs).strip()


# ---------------------------------------------------------------------------
# Voice command stop phrases
# ---------------------------------------------------------------------------

_STOP_PHRASES = {"stop", "stop listening", "stop recording", "pause recording"}


def _ends_with_stop_phrase(text: str) -> bool:
    t = text.lower().strip().rstrip(".,!?")
    for phrase in _STOP_PHRASES:
        if t.endswith(phrase):
            return True
    return False


def _strip_stop_phrase(text: str) -> str:
    t = text.strip()
    t_low = t.lower().rstrip(".,!?")
    for phrase in _STOP_PHRASES:
        if t_low.endswith(phrase):
            return t[: len(t) - len(phrase)].strip()
    return t


# ---------------------------------------------------------------------------
# StreamingRecorder
# ---------------------------------------------------------------------------


class StreamingRecorder:
    """Real-time STT: tiny whisper for display + full model for accurate final.

    No subprocess. Own VAD. Own stop control.

    Usage::

        rec = StreamingRecorder(on_partial=lambda t: ...)
        result, audio = rec.listen_push_to_talk()
    """

    def __init__(self, on_partial: Callable[[str], None] | None = None) -> None:
        self._on_partial = on_partial
        self._last_partial: str = ""
        self._last_partial_time: float = 0.0

    def listen(self) -> tuple[TranscriptResult, np.ndarray]:
        """Alias for listen_push_to_talk()."""
        return self.listen_push_to_talk()

    def listen_push_to_talk(self) -> tuple[TranscriptResult, np.ndarray]:
        """Record with real-time word display and accurate final transcript.

        Phases:
          1. Press SPACE to start recording.
          2. Tiny whisper runs every 300 ms — words appear as you speak.
          3. Stop triggers (first wins): VAD silence / SPACE / voice command.
          4. Full model re-transcribes the complete audio — accurate final displayed.

        GPU: ~150 MB during recording (tiny), ~2.15 GB peak during finalization (tiny + full).
        Both models freed on process exit.
        """
        from voice_agent.core.audio import _require_sounddevice, _wait_for_key
        from silero_vad import VADIterator

        sd = _require_sounddevice()
        tiny = _get_tiny_model()
        vad_model = _get_vad_model()

        print("Press SPACE to start recording...", flush=True)
        _wait_for_key(" ")
        print("Recording... (stops on silence, or press SPACE)", flush=True)

        # Shared state
        audio_queue: Queue = Queue()
        audio_frames: list[np.ndarray] = []
        speech_started = threading.Event()
        stop_event = threading.Event()
        self._last_partial = ""
        self._last_partial_time = 0.0

        # --- Audio callback (native audio thread — keep minimal) -----------
        def _audio_cb(indata, frames, t, status):
            audio_queue.put(indata[:, 0].copy())

        # --- VAD + accumulation thread -------------------------------------
        vad = VADIterator(
            vad_model,
            threshold=0.5,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=1200,
            speech_pad_ms=30,
        )

        def _vad_thread() -> None:
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio_frames.append(chunk)
                event = vad(chunk, return_seconds=False)
                if event and "start" in event:
                    speech_started.set()
                if event and "end" in event and speech_started.is_set():
                    stop_event.set()

        # --- Transcription thread (tiny model, every 300 ms) ---------------
        # Transcribes a sliding window of the last DISPLAY_WINDOW_SEC seconds.
        # - Keeps inference fast regardless of utterance length.
        # - Uses full line overwrite so corrections are always visible.
        DISPLAY_WINDOW_SEC = 5
        DISPLAY_WINDOW_CHUNKS = int(DISPLAY_WINDOW_SEC * SAMPLE_RATE / _VAD_CHUNK)

        def _transcription_thread() -> None:
            last_display = ""
            while not stop_event.is_set():
                time.sleep(0.3)
                if not speech_started.is_set() or not audio_frames:
                    continue

                # Slide: take only the most recent N chunks so tiny stays fast
                window = audio_frames[-DISPLAY_WINDOW_CHUNKS:]
                audio = np.concatenate(window)
                try:
                    segs, _ = tiny.transcribe(
                        audio, language="en", beam_size=1, best_of=1, vad_filter=False
                    )
                    text = "".join(s.text for s in segs).strip()
                except Exception:
                    continue

                if text and text != last_display:
                    # Full line overwrite — always shows whisper's current best guess.
                    # Corrections and new words both update immediately.
                    print(f"\r\033[K{text}", end="", flush=True)
                    last_display = text

                if text:
                    # _last_partial accumulates across windows for accurate final fallback
                    self._last_partial = text
                    self._last_partial_time = time.time()

                    if self._on_partial is not None:
                        self._on_partial(text)

                    if _ends_with_stop_phrase(text):
                        stop_event.set()

        threading.Thread(target=_vad_thread, daemon=True).start()
        threading.Thread(target=_transcription_thread, daemon=True).start()

        # --- Main thread: audio stream + SPACE poll ------------------------
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_VAD_CHUNK,
            callback=_audio_cb,
        ):
            if sys.platform == "win32":
                import msvcrt
                while not stop_event.is_set():
                    if msvcrt.kbhit() and msvcrt.getch() == b" ":
                        stop_event.set()
                        break
                    stop_event.wait(timeout=0.05)
            else:
                stop_event.wait()

        # Give transcription thread one last pass before finalizing
        time.sleep(0.35)

        # --- Accurate final pass (full model) ------------------------------
        audio_array = (
            np.concatenate(audio_frames) if audio_frames else np.zeros(512, dtype=np.float32)
        )

        partial_text = _strip_stop_phrase(self._last_partial)

        if audio_array.size > 512:
            print("\nFinalizing...", end="", flush=True)
            accurate_text = _transcribe_final_accurate(audio_array)
        else:
            accurate_text = ""

        final_text = accurate_text or partial_text
        print(f"\r\033[K{final_text}")

        wav_path = write_wav(audio_array)

        cfg = settings.load()
        skip_confidence = cfg["stt"].get("streaming_skip_confidence_pass", False)
        confidence = 1.0 if skip_confidence else 0.85

        result = TranscriptResult(
            text=final_text,
            language="en",
            confidence=confidence,
            segments=[],
            audio_path_temp=wav_path,
        )
        return result, audio_array

    def stop(self) -> None:
        """No-op: no subprocess to stop. Kept for API compatibility."""
