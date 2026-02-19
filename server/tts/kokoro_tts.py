"""Text-to-speech using Kokoro-82M (local neural TTS, Apache 2.0)."""

import logging
import re
from typing import Generator

import numpy as np

log = logging.getLogger(__name__)

_LANG_TO_KOKORO: dict[str, str] = {
    "en": "a", "en-us": "a", "en-gb": "b",
    "fr": "f", "es": "e", "it": "i",
    "hi": "h", "ja": "j", "zh": "z", "pt": "p",
}
_UNSUPPORTED_LANGS: set[str] = {"de"}
_SAMPLE_RATE = 24000


class KokoroTTS:
    """Synthesizes speech via the Kokoro-82M neural TTS model.

    Supports English and other languages with Kokoro lang_code mappings.
    Falls back to PiperTTS for languages not supported by Kokoro (e.g. German).
    """

    def __init__(self, tts_config: dict):
        from kokoro import KPipeline

        self._default_language = tts_config.get("default_language", "en")
        self._sentence_silence = tts_config.get("sentence_silence", 0.2)
        self._speed = float(tts_config.get("speed", 1.0))

        self._pipelines: dict[str, KPipeline] = {}   # lang_code -> KPipeline
        self._kokoro_voices: dict[str, str] = {}      # language  -> voice name

        voices_cfg: dict = tts_config.get("voices") or {}

        # Load Kokoro pipelines for supported languages
        for lang, voice_cfg in voices_cfg.items():
            kokoro_voice = voice_cfg.get("kokoro_voice")
            if not kokoro_voice:
                continue
            lang_code = _LANG_TO_KOKORO.get(lang.lower())
            if lang_code is None:
                log.warning("No Kokoro lang_code mapping for '%s' — skipping", lang)
                continue
            if lang_code not in self._pipelines:
                log.info("Loading Kokoro pipeline for lang_code='%s'", lang_code)
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
            self._kokoro_voices[lang] = kokoro_voice

        # Load Piper fallback for unsupported languages that have a piper_voice key
        piper_voices = {
            lang: cfg for lang, cfg in voices_cfg.items()
            if lang in _UNSUPPORTED_LANGS and cfg.get("piper_voice")
        }
        if piper_voices:
            from server.tts.piper_tts import PiperTTS
            piper_cfg = {**tts_config, "voices": piper_voices}
            self._piper_fallback: PiperTTS | None = PiperTTS(piper_cfg)
            log.info("Loaded Piper fallback for languages: %s", list(piper_voices))
        else:
            self._piper_fallback = None

        if not self._pipelines and not self._piper_fallback:
            raise RuntimeError("KokoroTTS: no voices configured — set kokoro_voice in tts.voices")

    def _resolve(self, language: str | None) -> tuple[object, str]:
        """Return (pipeline, voice_name) for the requested language."""
        lang = (language or self._default_language).lower()
        lang_code = _LANG_TO_KOKORO.get(lang)

        if lang_code and lang_code in self._pipelines and lang in self._kokoro_voices:
            return self._pipelines[lang_code], self._kokoro_voices[lang]

        # Fall back to default language
        default = self._default_language.lower()
        default_code = _LANG_TO_KOKORO.get(default)
        if default_code and default_code in self._pipelines:
            voice = self._kokoro_voices.get(default, next(iter(self._kokoro_voices.values())))
            return self._pipelines[default_code], voice

        # Last resort: pick any available pipeline
        code = next(iter(self._pipelines))
        voice = next(iter(self._kokoro_voices.values()))
        return self._pipelines[code], voice

    def synthesize_chunks(
        self,
        text: str,
        language: str | None = None,
    ) -> Generator[tuple[np.ndarray, int, bool], None, None]:
        """Yield per-sentence TTS audio chunks for streaming to client.

        Yields ``(audio_int16, sample_rate, is_last)`` tuples.
        Audio is int16 PCM at 24000 Hz (matching the wire protocol).

        Delegates unsupported languages (e.g. German) to the Piper fallback.
        """
        lang = (language or self._default_language).lower()

        if lang in _UNSUPPORTED_LANGS:
            if self._piper_fallback is not None:
                yield from self._piper_fallback.synthesize_chunks(text, language=lang)
            else:
                # No Piper fallback — route to any available Kokoro voice.
                # Do NOT recurse through synthesize_chunks with self._default_language:
                # if default_language is also unsupported (e.g. "de"), that would
                # recurse indefinitely.  self._kokoro_voices only contains languages
                # with a valid Kokoro mapping, so iterating it is always safe.
                if not self._kokoro_voices:
                    log.error(
                        "Language '%s' not supported by Kokoro and no fallback available", lang
                    )
                    return
                kokoro_lang = next(iter(self._kokoro_voices))
                log.warning(
                    "Language '%s' not supported by Kokoro and no Piper fallback configured "
                    "— using Kokoro voice for '%s' instead",
                    lang, kokoro_lang,
                )
                yield from self.synthesize_chunks(text, language=kokoro_lang)
            return

        pipeline, voice_name = self._resolve(lang)
        sentences = _split_sentences(text)
        if not sentences:
            return

        for i, sentence in enumerate(sentences):
            is_last = i == len(sentences) - 1

            arrays: list[np.ndarray] = []
            for _gs, _ps, audio in pipeline(sentence.strip(), voice=voice_name, speed=self._speed):
                if audio is not None and len(audio) > 0:
                    arrays.append(audio)

            if not arrays:
                if is_last:
                    yield np.array([], dtype=np.int16), _SAMPLE_RATE, True
                continue

            audio_f32 = np.concatenate(arrays)

            # Add inter-sentence silence (except after the last sentence)
            if not is_last:
                silence_samples = int(self._sentence_silence * _SAMPLE_RATE)
                silence = np.zeros(silence_samples, dtype=np.float32)
                audio_f32 = np.concatenate([audio_f32, silence])

            # Kokoro's neural vocoder can slightly exceed [-1, 1] — clip before scaling
            audio_int16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
            yield audio_int16, _SAMPLE_RATE, is_last

    def synthesize(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """Convert *text* to audio using the voice for *language*.

        Returns ``(audio_float32, sample_rate)``.
        """
        arrays: list[np.ndarray] = []
        sample_rate = _SAMPLE_RATE  # default; overwritten by actual chunk sample rate
        for audio_int16, sr, _is_last in self.synthesize_chunks(text, language=language):
            sample_rate = sr
            if len(audio_int16) > 0:
                arrays.append(audio_int16.astype(np.float32) / 32767.0)
        audio_f32 = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
        return audio_f32, sample_rate


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]
