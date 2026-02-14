"""TTS engine abstraction — protocol + factory."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSEngine(Protocol):
    """Any TTS backend must implement this interface."""

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert *text* to audio.

        Returns ``(audio_float32, sample_rate)``.
        """
        ...


def create_tts(tts_config: dict) -> TTSEngine:
    """Instantiate the TTS engine specified by ``tts_config["engine"]``.

    Falls back to macOS ``say`` when the key is absent or set to ``"say"``.
    """
    engine = tts_config.get("engine", "say")

    if engine == "piper":
        from tts.piper_tts import PiperTTS

        return PiperTTS(tts_config)

    # Default: macOS say
    from tts.mac_say import MacTTS

    return MacTTS(tts_config)
