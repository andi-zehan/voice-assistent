"""Text-to-speech using macOS `say` command."""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


class MacTTS:
    """Synthesizes speech via macOS `say -o` → AIFF file → numpy array.

    Writing to a temp file then reading it back lets us use sounddevice for
    playback, which supports instant stop() for barge-in.
    """

    def __init__(self, tts_config: dict):
        self._voice = tts_config["voice"]
        self._rate = tts_config["rate"]
        self._output_sample_rate = tts_config.get("output_sample_rate", 22050)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert text to audio.

        Returns (audio_float32, sample_rate).
        """
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp_path = f.name

        try:
            subprocess.run(
                [
                    "say",
                    "-v", self._voice,
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
