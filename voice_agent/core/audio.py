# audio.py — microphone capture with silero-VAD end-of-speech detection.
#
# Public API:
#   SAMPLE_RATE            — 16000 Hz (required by both Whisper and silero-VAD)
#   record_until_silence() — record from mic, stop when speech ends, return (array, wav_path)
#   write_wav(audio)       — write float32 mono 16 kHz array to a temp WAV file, return path

import sys
import tempfile
import threading
import wave

import numpy as np

SAMPLE_RATE = 16000
_VAD_CHUNK = 512  # silero-VAD requires exactly 512 samples at 16 kHz

_vad_model = None  # module-level singleton, loaded once on first call


def record_until_silence(
    silence_ms: int = 1500,
    max_sec: float = 30.0,
    start_threshold: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Record from the default microphone until silence_ms of silence follows speech.

    Blocks until the user stops speaking or max_sec is reached.

    Returns:
        audio  — float32 numpy array at 16 kHz mono
        path   — path to a temp WAV file (same audio, written for Phase 3 storage)

    The temp file is owned by the caller; delete it when no longer needed.
    Phase 3 moves it to ~/.voice_agent/pending/ before deletion.
    """
    sd = _require_sounddevice()
    VADIterator = _require_vad_iterator()

    model = _get_vad_model()
    vad = VADIterator(
        model,
        threshold=start_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=silence_ms,
        speech_pad_ms=30,
    )

    max_chunks = int(max_sec * SAMPLE_RATE / _VAD_CHUNK)
    frames: list[np.ndarray] = []
    speech_started = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_VAD_CHUNK,
    ) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(_VAD_CHUNK)
            chunk: np.ndarray = data[:, 0]  # take channel 0, shape (512,)

            event = vad(chunk, return_seconds=False)

            if event and "start" in event:
                speech_started = True

            if speech_started:
                frames.append(chunk.copy())

            if event and "end" in event and speech_started:
                break

    audio = np.concatenate(frames) if frames else np.zeros(_VAD_CHUNK, dtype=np.float32)
    wav_path = write_wav(audio)
    return audio, wav_path


def record_with_key_toggle(key: str = " ") -> tuple[np.ndarray, str]:
    """Record audio with push-to-talk: SPACE starts, VAD auto-stops or SPACE stops.

    Background thread records continuously. VAD detects end-of-speech and sets the
    stop event automatically. Main thread polls for SPACE as a manual backup using
    msvcrt.kbhit() (non-blocking, no subprocess conflicts on Windows).
    """
    import sys

    sd = _require_sounddevice()
    VADIterator = _require_vad_iterator()
    vad_model = _get_vad_model()

    print("Press SPACE to start recording...", flush=True)
    _wait_for_key(key)
    print("Recording... (stops on silence, or press SPACE)", flush=True)

    vad = VADIterator(
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=1200,
        speech_pad_ms=30,
    )

    stop_event = threading.Event()
    speech_started = [False]
    frames: list[np.ndarray] = []

    def _record() -> None:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_VAD_CHUNK,
        ) as stream:
            while not stop_event.is_set():
                data, _ = stream.read(_VAD_CHUNK)
                chunk = data[:, 0]
                frames.append(chunk.copy())

                event = vad(chunk, return_seconds=False)
                if event and "start" in event:
                    speech_started[0] = True
                if event and "end" in event and speech_started[0]:
                    stop_event.set()

    record_thread = threading.Thread(target=_record, daemon=True)
    record_thread.start()

    # Main thread: poll SPACE key without blocking audio thread
    if sys.platform == "win32":
        import msvcrt
        while not stop_event.is_set():
            if msvcrt.kbhit() and msvcrt.getch() == b" ":
                stop_event.set()
                break
            stop_event.wait(timeout=0.05)
    else:
        _wait_for_key(key)
        stop_event.set()

    record_thread.join(timeout=2.0)

    audio = np.concatenate(frames) if frames else np.zeros(_VAD_CHUNK, dtype=np.float32)
    return audio, write_wav(audio)


def _wait_for_key(key: str = " ") -> None:
    """Block until `key` is pressed. No extra dependencies — uses msvcrt on Windows, tty on Unix."""
    if sys.platform == "win32":
        import msvcrt
        target = key.encode()
        while msvcrt.getch() != target:
            pass
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while sys.stdin.read(1) != key:
                pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def write_wav(audio: np.ndarray) -> str:
    """Write float32 mono 16 kHz array to a temp WAV file. Return the file path."""
    clamped = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (clamped * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())

    return tmp.name


# --- lazy imports with friendly error messages ---

def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            raise RuntimeError(
                "silero-vad is not installed. Run: pip install silero-vad"
            )
        _vad_model = load_silero_vad()
    return _vad_model


def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install sounddevice"
        )


def _require_vad_iterator():
    try:
        from silero_vad import VADIterator
        return VADIterator
    except ImportError:
        raise RuntimeError(
            "silero-vad is not installed. Run: pip install silero-vad"
        )
