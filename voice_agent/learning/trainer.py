# trainer.py — LoRA fine-tune scheduler and background subprocess launcher.
#
# Public API (called by rl_loop and cli.py):
#   maybe_trigger() — check pending count; spawn training subprocess if threshold met
#   run_now()       — force training immediately regardless of count
#
# The actual training (_run_training) runs in a separate low-priority subprocess
# so the CLI remains responsive during fine-tuning (D003).
#
# When invoked as __main__ (i.e. by the subprocess), it calls _run_training().

from __future__ import annotations

import sys

from voice_agent.config import settings
from voice_agent.learning import store


def maybe_trigger() -> None:
    """Spawn a training subprocess if pending count meets min_samples_to_train (D007)."""
    cfg = settings.load()
    n = store.count_pending()
    if n >= cfg["rl"]["min_samples_to_train"]:
        _spawn()
        print(f"[trainer] Fine-tune started in background ({n} samples)", file=sys.stderr)


def run_now() -> None:
    """Force a training run immediately, regardless of pending count."""
    n = store.count_pending()
    if n == 0:
        print("[trainer] No pending corrections — nothing to train on.", file=sys.stderr)
        return
    _spawn()
    print(f"[trainer] Fine-tune started in background ({n} samples)", file=sys.stderr)


def _spawn() -> None:
    """Launch _run_training() as a detached low-priority subprocess."""
    import subprocess

    kwargs: dict = {
        "args": [sys.executable, "-m", "voice_agent.learning.trainer"],
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS

    subprocess.Popen(**kwargs)


# ---------------------------------------------------------------------------
# Training subprocess entry point
# ---------------------------------------------------------------------------

def _run_training() -> None:
    """LoRA fine-tune on pending correction pairs. Runs in subprocess.

    Uses HuggingFace transformers + PEFT so the base faster-whisper model
    weights are never modified (D003). Saves adapter to ADAPTER_DIR and
    deletes pending audio (D005).
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch
    except ImportError as e:
        print(f"[trainer] Missing dep: {e}. Run: pip install transformers peft", file=sys.stderr)
        return

    import wave
    import numpy as np
    from voice_agent.core.hardware import detect

    pairs = store.load_pending()
    if not pairs:
        return

    cfg = settings.load()
    info = detect()
    raw_model = cfg["stt"]["model"] or info.recommended_stt
    hf_name = _to_hf_name(raw_model)

    print(f"[trainer] Loading {hf_name} for fine-tuning...", file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(hf_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(hf_name)

    adapter_dir = store.ADAPTER_DIR / "whisper_lora"

    if adapter_dir.exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        print("[trainer] Continuing from existing adapter.", file=sys.stderr)
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(base_model, lora_cfg)

    device = "cuda" if info.cuda_available else "cpu"
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[trainer] Training on {len(pairs)} correction(s)...", file=sys.stderr)

    for pair in pairs:
        audio = _load_wav(pair["audio_path"])
        if audio is None:
            continue

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        labels = processor.tokenizer(
            pair["corrected"], return_tensors="pt"
        ).input_ids.to(device)

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    store.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    print(f"[trainer] Adapter saved to {adapter_dir}", file=sys.stderr)

    store.clear_pending()
    print("[trainer] Pending audio cleared (D005).", file=sys.stderr)


def _to_hf_name(model_name: str) -> str:
    """Map faster-whisper checkpoint names to HuggingFace model IDs."""
    _MAP = {
        "tiny":     "openai/whisper-tiny",
        "base":     "openai/whisper-base",
        "medium":   "openai/whisper-medium",
        "large-v3": "openai/whisper-large-v3",
    }
    return _MAP.get(model_name, f"openai/whisper-{model_name}")


def _load_wav(path: str) -> "np.ndarray | None":
    """Read a 16-bit mono WAV file and return a float32 numpy array."""
    import wave
    import numpy as np

    try:
        with wave.open(path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        return pcm
    except Exception as e:
        print(f"[trainer] Could not read {path}: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    _run_training()
