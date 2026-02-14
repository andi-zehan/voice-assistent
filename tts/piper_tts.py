"""Text-to-speech using Piper (local neural TTS)."""

from pathlib import Path

import numpy as np
from piper.voice import PiperVoice


class PiperTTS:
    """Synthesizes speech via a local Piper ONNX voice model.

    The model is loaded once at init; subsequent ``synthesize()`` calls reuse it.
    """

    def __init__(self, tts_config: dict):
        model_dir = Path(tts_config.get("model_dir", "models/piper"))
        voice_name = tts_config.get("piper_voice", "en_US-lessac-medium")
        self._sentence_silence = tts_config.get("sentence_silence", 0.2)
        self._length_scale = tts_config.get("length_scale")
        self._noise_scale = tts_config.get("noise_scale")
        self._noise_w_scale = tts_config.get("noise_w_scale")

        model_path = model_dir / f"{voice_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}\n"
                "Download it from: https://huggingface.co/rhasspy/piper-voices\n"
                f"  curl -L -o {model_path} "
                f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice_name}.onnx\n"
                f"  curl -L -o {model_path}.json "
                f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{voice_name}.onnx.json"
            )

        self._voice = PiperVoice.load(str(model_path))
        self._sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert *text* to audio.

        Returns ``(audio_float32, sample_rate)``.
        """
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w_scale,
        )

        silence_samples = int(self._sentence_silence * self._sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)

        arrays: list[np.ndarray] = []
        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            if arrays:
                arrays.append(silence)
            arrays.append(chunk.audio_float_array)

        audio = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
        return audio, self._sample_rate
