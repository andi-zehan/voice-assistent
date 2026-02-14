"""Text-to-speech using Piper (local neural TTS)."""

import logging
import re
from pathlib import Path
from typing import Generator

import numpy as np
from piper.voice import PiperVoice

log = logging.getLogger(__name__)


class PiperTTS:
    """Synthesizes speech via local Piper ONNX voice models.

    When ``voices`` is present in the config, one model per language is loaded
    at init time.  ``synthesize()`` selects the voice matching the requested
    language, falling back to ``default_language``.
    """

    def __init__(self, tts_config: dict):
        model_dir = Path(tts_config.get("model_dir", "models/piper"))
        self._sentence_silence = tts_config.get("sentence_silence", 0.2)
        self._length_scale = tts_config.get("length_scale")
        self._noise_scale = tts_config.get("noise_scale")
        self._noise_w_scale = tts_config.get("noise_w_scale")

        self._default_language = tts_config.get("default_language", "en")
        self._voices: dict[str, PiperVoice] = {}
        self._sample_rates: dict[str, int] = {}

        voices_cfg = tts_config.get("voices")
        if voices_cfg:
            for lang, voice_cfg in voices_cfg.items():
                voice_name = voice_cfg.get("piper_voice")
                if not voice_name:
                    continue
                model_path = model_dir / f"{voice_name}.onnx"
                if not model_path.exists():
                    log.warning("Piper voice model not found for '%s': %s — skipping", lang, model_path)
                    continue
                voice = PiperVoice.load(str(model_path))
                self._voices[lang] = voice
                self._sample_rates[lang] = voice.config.sample_rate
                log.info("Loaded Piper voice for '%s': %s", lang, voice_name)
        else:
            voice_name = tts_config.get("piper_voice", "en_US-lessac-medium")
            model_path = model_dir / f"{voice_name}.onnx"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Piper voice model not found: {model_path}\n"
                    "Download it from: https://huggingface.co/rhasspy/piper-voices"
                )
            voice = PiperVoice.load(str(model_path))
            self._voices[self._default_language] = voice
            self._sample_rates[self._default_language] = voice.config.sample_rate

        if not self._voices:
            raise FileNotFoundError("No Piper voice models could be loaded")

    def _resolve_voice(self, language: str | None = None) -> tuple[PiperVoice, int, str]:
        """Resolve voice, sample_rate, and effective language."""
        lang = language if language and language in self._voices else self._default_language
        return self._voices[lang], self._sample_rates[lang], lang

    def synthesize(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """Convert *text* to audio using the voice for *language*.

        Returns ``(audio_float32, sample_rate)``.
        """
        from piper.config import SynthesisConfig

        voice, sample_rate, _ = self._resolve_voice(language)

        syn_config = SynthesisConfig(
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w_scale,
        )

        silence_samples = int(self._sentence_silence * sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)

        arrays: list[np.ndarray] = []
        for chunk in voice.synthesize(text, syn_config=syn_config):
            if arrays:
                arrays.append(silence)
            arrays.append(chunk.audio_float_array)

        audio = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
        return audio, sample_rate

    def synthesize_chunks(
        self,
        text: str,
        language: str | None = None,
    ) -> Generator[tuple[np.ndarray, int, bool], None, None]:
        """Yield per-sentence TTS audio chunks for streaming to client.

        Splits *text* into sentences and synthesizes each independently.
        Yields ``(audio_int16, sample_rate, is_last)`` tuples.

        Audio is returned as int16 PCM (matching the wire protocol) rather
        than float32.
        """
        from piper.config import SynthesisConfig

        voice, sample_rate, _ = self._resolve_voice(language)

        syn_config = SynthesisConfig(
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w_scale,
        )

        sentences = _split_sentences(text)
        if not sentences:
            return

        for i, sentence in enumerate(sentences):
            is_last = i == len(sentences) - 1

            # Synthesize this sentence
            arrays: list[np.ndarray] = []
            for chunk in voice.synthesize(sentence.strip(), syn_config=syn_config):
                arrays.append(chunk.audio_float_array)

            if not arrays:
                if is_last:
                    yield np.array([], dtype=np.int16), sample_rate, True
                continue

            audio_f32 = np.concatenate(arrays)

            # Add inter-sentence silence (except after the last sentence)
            if not is_last:
                silence_samples = int(self._sentence_silence * sample_rate)
                silence = np.zeros(silence_samples, dtype=np.float32)
                audio_f32 = np.concatenate([audio_f32, silence])

            # Convert to int16 for wire protocol
            audio_int16 = (audio_f32 * 32767).astype(np.int16)
            yield audio_int16, sample_rate, is_last


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    # Split on sentence boundaries: . ! ? followed by space or end-of-string
    # Keep the punctuation with the preceding sentence
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]
