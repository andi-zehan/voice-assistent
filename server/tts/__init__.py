"""TTS engine abstraction — Piper only (server is Linux)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSEngine(Protocol):
    """Any TTS backend must implement this interface."""

    def synthesize(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """Convert *text* to audio.

        Returns ``(audio_float32, sample_rate)``.
        """
        ...


def create_tts(tts_config: dict) -> TTSEngine:
    """Instantiate the Piper TTS engine."""
    from server.tts.piper_tts import PiperTTS

    return PiperTTS(tts_config)
