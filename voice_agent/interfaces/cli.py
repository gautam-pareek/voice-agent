from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="voice",
    help="voice-agent: local-first STT + TTS",
    no_args_is_help=True,
)


@app.command()
def setup(
    detect: bool = typer.Option(False, "--detect", help="Print hardware tier and recommended models (read-only, no download)"),
    tier: Optional[int] = typer.Option(None, "--tier", min=1, max=4, help="Override auto-detected tier (1-4)"),
) -> None:
    """Detect hardware tier and download the right AI models. Run once before voice listen."""
    from voice_agent.core.hardware import detect as hw_detect, ensure_ready
    from voice_agent.config import settings

    if tier is not None:
        cfg = settings.load()
        cfg["tier"] = tier
        settings.save(cfg)
        print(f"Tier overridden to {tier}. Run 'voice setup' to download models for this tier.", flush=True)
        return

    ensure_ready()
    info = hw_detect()

    print(f"Hardware Tier:     {info.tier}", flush=True)
    print(f"GPU detected:      {info.cuda_available}", flush=True)
    print(f"GPU backend:       {info.backend}", flush=True)
    print(f"VRAM:              {info.vram_gb:.1f} GB", flush=True)
    print(f"Torch ready:       {info.torch_ready}", flush=True)
    print(f"CPU cores:         {info.cpu_cores}", flush=True)
    print(f"CPU threads:       {info.cpu_threads}  (used for inference)", flush=True)
    print(f"RAM:               {info.ram_gb:.1f} GB", flush=True)
    print(f"Recommended STT:   whisper-{info.recommended_stt}", flush=True)
    print(f"Recommended TTS:   {info.recommended_tts}", flush=True)

    if detect:
        return  # --detect is read-only: show info but don't download

    print("", flush=True)
    _install_stt_model(info)
    print("", flush=True)
    print("Setup complete. Run 'voice listen' to start transcribing.", flush=True)


def _install_stt_model(info) -> None:
    """Download and cache the STT model for the detected tier, plus the tiny streaming model."""
    from faster_whisper import WhisperModel
    from voice_agent.config import settings

    model_name = info.recommended_stt
    device = "cuda" if info.torch_ready else "cpu"
    compute_type = "int8_float16" if info.torch_ready else "int8"

    print(f"Downloading STT model: whisper-{model_name} ({device}, {compute_type})...", flush=True)
    print("(This is a one-time download. Progress shown below.)", flush=True)

    try:
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        model_cache_name = f"models--Systran--faster-whisper-{model_name}"
        already_cached = os.path.isdir(os.path.join(cache_dir, model_cache_name))
        WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=already_cached)
        cfg = settings.load()
        cfg["stt"]["model"] = model_name
        settings.save(cfg)
        print(f"whisper-{model_name} ready.", flush=True)
    except Exception as e:
        print(f"Download failed: {e}", flush=True)
        print(f"Retry manually: pip install faster-whisper, then run 'voice setup' again.", flush=True)
        return

    # Also pre-download tiny model used for real-time streaming partials (D020).
    # Always uses CPU/int8 regardless of tier — tiny runs fast on any hardware.
    if model_name != "tiny":
        try:
            tiny_cache = os.path.join(cache_dir, "models--Systran--faster-whisper-tiny")
            tiny_cached = os.path.isdir(tiny_cache)
            if not tiny_cached:
                print("Downloading whisper-tiny (streaming partial model)...", flush=True)
            WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=tiny_cached)
            print("whisper-tiny ready.", flush=True)
        except Exception as e:
            print(f"whisper-tiny download failed (streaming will download on first use): {e}", flush=True)


@app.command()
def listen(
    watch: bool = typer.Option(False, "--watch", help="Continuous mode: keep listening after each sentence"),
    type_: bool = typer.Option(False, "--type", help="Type transcription into focused window"),
    clipboard: bool = typer.Option(False, "--clipboard", help="Copy transcription to clipboard"),
    batch: bool = typer.Option(False, "--batch", help="Batch mode: record then transcribe (no real-time display)"),
) -> None:
    """Transcribe speech with real-time word display. Press SPACE to start; stops on silence, SPACE, or 'stop listening'."""
    if batch:
        _listen_batch(watch, type_, clipboard, push_to_talk=True)
    else:
        _listen_streaming(watch, type_, clipboard)


def _listen_batch(watch: bool, type_: bool, clipboard: bool, push_to_talk: bool) -> None:
    """Batch mode: record until silence, then transcribe."""
    import os
    from voice_agent.core import audio, stt

    def _run_once() -> None:
        if push_to_talk:
            audio_array, tmp_path = audio.record_with_key_toggle()
        else:
            print("Listening... (speak now)", flush=True)
            audio_array, tmp_path = audio.record_until_silence()

        print("Transcribing...", flush=True)
        result = stt.transcribe(
            audio_array,
            audio_path=tmp_path,
            on_segment=lambda t: print(t, end="", flush=True),
        )
        print()  # newline after streamed segment output

        if clipboard:
            import pyperclip
            pyperclip.copy(result["text"])
            print("(copied to clipboard)", flush=True)

        if type_:
            import time
            import pyautogui
            time.sleep(0.3)  # give user time to focus the target window
            pyautogui.typewrite(result["text"], interval=0.02)

        from voice_agent.learning import rl_loop
        rl_loop.process(result, audio_array)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if watch:
        print("Continuous mode — press Ctrl+C to stop.", flush=True)
        try:
            while True:
                _run_once()
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
    else:
        _run_once()


def _listen_streaming(watch: bool, type_: bool, clipboard: bool) -> None:
    """Streaming mode: partials display as words are spoken, accurate final on silence."""
    import os
    from voice_agent.core.streaming_stt import StreamingRecorder
    from voice_agent.learning import rl_loop

    last_typed: list[str] = [""]  # list so nested function can mutate it

    def _on_partial(text: str) -> None:
        """Delta-type new characters into the focused window as partials arrive."""
        if not type_:
            return
        import pyautogui
        prev = last_typed[0]
        if text.startswith(prev):
            new_chars = text[len(prev):]
            if new_chars:
                pyautogui.typewrite(new_chars, interval=0.02)
                last_typed[0] = text

    def _run_once() -> None:
        last_typed[0] = ""
        recorder = StreamingRecorder(on_partial=_on_partial)
        result, audio_array = recorder.listen_push_to_talk()

        if clipboard:
            import pyperclip
            pyperclip.copy(result["text"])
            print("(copied to clipboard)", flush=True)

        if type_:
            # Type any remaining characters the partials didn't cover
            import pyautogui
            typed_so_far = last_typed[0]
            final = result["text"]
            if final.startswith(typed_so_far):
                remaining = final[len(typed_so_far):]
            else:
                remaining = final  # model revised heavily — type full final
            if remaining:
                pyautogui.typewrite(remaining, interval=0.02)

        rl_loop.process(result, audio_array)

        try:
            if result["audio_path_temp"]:
                os.unlink(result["audio_path_temp"])
        except OSError:
            pass

    if watch:
        print("Continuous streaming mode — press Ctrl+C to stop.", flush=True)
        try:
            while True:
                _run_once()
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
    else:
        _run_once()


@app.command()
def transcribe(
    file: Path = typer.Argument(..., help="Audio file to transcribe (.wav, .mp3, .m4a, .ogg, ...)"),
    clipboard: bool = typer.Option(False, "--clipboard", help="Copy result to clipboard"),
    type_: bool = typer.Option(False, "--type", help="Type result into focused window"),
) -> None:
    """Transcribe a pre-recorded audio file through the same pipeline as voice listen."""
    import os

    import numpy as np

    if not file.exists():
        typer.echo(f"Error: file not found: {file}", err=True)
        raise typer.Exit(1)

    audio_array = _load_audio_file(file)

    from voice_agent.core import stt
    from voice_agent.core.audio import write_wav

    wav_path = write_wav(audio_array)

    print(f"Transcribing {file.name}...", flush=True)
    result = stt.transcribe(
        audio_array,
        audio_path=wav_path,
        on_segment=lambda t: print(t, end="", flush=True),
    )
    print()

    if clipboard:
        import pyperclip
        pyperclip.copy(result["text"])
        print("(copied to clipboard)", flush=True)

    if type_:
        import time
        import pyautogui
        time.sleep(0.3)
        pyautogui.typewrite(result["text"], interval=0.02)

    from voice_agent.learning import rl_loop
    rl_loop.process(result, audio_array)

    try:
        os.unlink(wav_path)
    except OSError:
        pass


def _load_audio_file(file: Path) -> "np.ndarray":
    """Load any audio file, resample to 16 kHz mono float32."""
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")

    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise RuntimeError("scipy is not installed. Run: pip install scipy")

    from voice_agent.core.audio import SAMPLE_RATE

    data, sr = sf.read(str(file), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono

    if sr != SAMPLE_RATE:
        n_samples = int(len(data) * SAMPLE_RATE / sr)
        data = scipy_signal.resample(data, n_samples).astype(np.float32)

    return data.astype(np.float32)


@app.command()
def speak(text: str) -> None:
    """Speak text using the configured TTS engine."""
    typer.echo("TTS not yet implemented (Phase 4).")


@app.command()
def status() -> None:
    """Show current tier, models, pending samples, and last fine-tune date."""
    from voice_agent.core.hardware import detect as hw_detect
    from voice_agent.config import settings

    cfg = settings.load()
    info = hw_detect()
    typer.echo(f"Tier:              {info.tier}")
    typer.echo(f"STT model:         {cfg['stt']['model'] or 'not configured'}")
    typer.echo(f"TTS engine:        {cfg['tts']['engine'] or 'not configured'}")


@app.command()
def retrain() -> None:
    """Manually trigger LoRA fine-tune from pending corrections."""
    from voice_agent.learning import trainer
    trainer.run_now()


@app.command()
def mcp() -> None:
    """Start the MCP server for Claude Code / Cursor integration."""
    typer.echo("MCP server not yet implemented (Phase 7).")


packs_app = typer.Typer(help="Manage Piper TTS voice packs.")
app.add_typer(packs_app, name="packs")


@packs_app.command("list")
def packs_list() -> None:
    """List available Piper voice packs."""
    typer.echo("Pack manager not yet implemented (Phase 4).")


@packs_app.command("download")
def packs_download(name: str) -> None:
    """Download a Piper voice pack."""
    typer.echo("Pack manager not yet implemented (Phase 4).")


@packs_app.command("switch")
def packs_switch(name: str) -> None:
    """Switch the active Piper voice pack."""
    typer.echo("Pack manager not yet implemented (Phase 4).")
