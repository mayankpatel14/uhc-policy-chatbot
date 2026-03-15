"""Text-to-speech via Kokoro ONNX (82M parameter model, runs on CPU)."""

import io
import numpy as np

_kokoro_instance = None


def _get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is not None:
        return _kokoro_instance

    from huggingface_hub import hf_hub_download
    from kokoro_onnx import Kokoro

    model_path = hf_hub_download(
        repo_id="fastrtc/kokoro-onnx", filename="kokoro-v1.0.onnx"
    )
    voices_path = hf_hub_download(
        repo_id="fastrtc/kokoro-onnx", filename="voices-v1.0.bin"
    )
    _kokoro_instance = Kokoro(model_path, voices_path)
    return _kokoro_instance


def synthesize(text: str, voice: str = "af_sarah", speed: float = 1.0) -> bytes:
    """Convert text to WAV audio bytes."""
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()
