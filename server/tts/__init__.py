"""TTS engine abstraction — Piper and Kokoro supported."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass


@runtime_checkable
class TTSEngine(Protocol):
    """Any TTS backend must implement this interface."""

    def synthesize(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """Convert *text* to audio.

        Returns ``(audio_float32, sample_rate)``.
        """
        ...

    def synthesize_chunks(
        self, text: str, language: str | None = None
    ) -> "Generator[tuple[np.ndarray, int, bool], None, None]":
        """Yield per-sentence audio chunks for streaming.

        Yields ``(audio_int16, sample_rate, is_last)`` tuples.
        """
        ...


def create_tts(tts_config: dict) -> TTSEngine:
    """Instantiate the configured TTS engine (Piper or Kokoro)."""
    engine = tts_config.get("engine", "piper")
    if engine == "kokoro":
        from server.tts.kokoro_tts import KokoroTTS
        return KokoroTTS(tts_config)
    elif engine == "piper":
        from server.tts.piper_tts import PiperTTS
        return PiperTTS(tts_config)
    raise ValueError(f"Unknown TTS engine '{engine}'. Supported: 'piper', 'kokoro'.")
