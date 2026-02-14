"""Text-to-speech using macOS `say` command."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class MacTTS:
    """Synthesizes speech via macOS `say -o` → AIFF file → numpy array.

    Writing to a temp file then reading it back lets us use sounddevice for
    playback, which supports instant stop() for barge-in.
    """

    def __init__(self, tts_config: dict):
        self._rate = tts_config["rate"]
        self._output_sample_rate = tts_config.get("output_sample_rate", 22050)
        self._default_language = tts_config.get("default_language", "en")

        # Build language → voice mapping from voices config
        self._voice_map: dict[str, str] = {}
        voices_cfg = tts_config.get("voices")
        if voices_cfg:
            for lang, voice_cfg in voices_cfg.items():
                say_voice = voice_cfg.get("say_voice")
                if say_voice:
                    self._voice_map[lang] = say_voice
        else:
            # Backward compat: flat voice key
            self._voice_map[self._default_language] = tts_config["voice"]

    def synthesize(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """Convert text to audio using the voice for *language*.

        Returns (audio_float32, sample_rate).
        """
        lang = language if language and language in self._voice_map else self._default_language
        voice = self._voice_map[lang]

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp_path = f.name

        try:
            subprocess.run(
                [
                    "say",
                    "-v", voice,
                    "-r", str(self._rate),
                    "-o", tmp_path,
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )

            audio, sr = sf.read(tmp_path, dtype="float32")

            # Ensure mono
            if audio.ndim > 1:
                audio = audio[:, 0]

            return audio, sr

        finally:
            Path(tmp_path).unlink(missing_ok=True)
