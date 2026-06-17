from typing import Optional

import typer

app = typer.Typer(
    name="voice",
    help="voice-agent: local-first STT + TTS",
    no_args_is_help=True,
)


@app.command()
def setup(
    detect: bool = typer.Option(False, "--detect", help="Print hardware tier and recommended models"),
    tier: Optional[int] = typer.Option(None, "--tier", min=1, max=4, help="Override auto-detected tier (1-4)"),
) -> None:
    """One-time setup: record voice profile and configure hardware tier."""
    from voice_agent.core.hardware import detect as hw_detect
    from voice_agent.config import settings

    if detect:
        from voice_agent.core.hardware import ensure_ready
        ensure_ready()
        info = hw_detect()
        typer.echo(f"Hardware Tier:     {info.tier}")
        typer.echo(f"GPU detected:      {info.cuda_available}")
        typer.echo(f"GPU backend:       {info.backend}")
        typer.echo(f"VRAM:              {info.vram_gb:.1f} GB")
        typer.echo(f"Torch ready:       {info.torch_ready}")
        typer.echo(f"CPU cores:         {info.cpu_cores}")
        typer.echo(f"CPU threads:       {info.cpu_threads}  (used for inference)")
        typer.echo(f"RAM:               {info.ram_gb:.1f} GB")
        typer.echo(f"Recommended STT:   whisper-{info.recommended_stt}")
        typer.echo(f"Recommended TTS:   {info.recommended_tts}")
        return

    if tier is not None:
        cfg = settings.load()
        cfg["tier"] = tier
        settings.save(cfg)
        typer.echo(f"Tier overridden to {tier}. Run 'voice setup' to complete voice profile setup.")
        return

    typer.echo("Full setup wizard coming in Phase 5. Use --detect to check hardware.")


@app.command()
def listen(
    watch: bool = typer.Option(False, "--watch", help="Continuous transcription mode"),
    type_: bool = typer.Option(False, "--type", help="Type transcription into focused window"),
    clipboard: bool = typer.Option(False, "--clipboard", help="Copy transcription to clipboard"),
) -> None:
    """Transcribe speech from microphone."""
    import os
    from voice_agent.core import audio, stt

    def _run_once() -> None:
        typer.echo("Listening... (speak now)", err=True)
        audio_array, tmp_path = audio.record_until_silence()

        typer.echo("Transcribing...", err=True)
        result = stt.transcribe(
            audio_array,
            audio_path=tmp_path,
            on_segment=lambda t: typer.echo(t, nl=False),
        )
        typer.echo()  # newline after streamed segment output

        if clipboard:
            import pyperclip
            pyperclip.copy(result["text"])
            typer.echo("(copied to clipboard)", err=True)

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
        typer.echo("Continuous mode — press Ctrl+C to stop.", err=True)
        try:
            while True:
                _run_once()
        except KeyboardInterrupt:
            typer.echo("\nStopped.", err=True)
    else:
        _run_once()


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
