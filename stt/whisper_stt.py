"""Speech-to-text using faster-whisper."""

import time
import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    """Loads a Whisper model once and transcribes int16 audio buffers."""

    def __init__(self, stt_config: dict):
        self._model = WhisperModel(
            stt_config["model_size"],
            device=stt_config["device"],
            compute_type=stt_config["compute_type"],
        )
        self._language = stt_config.get("language")

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe an int16 audio buffer.

        Returns dict with keys: text, language, duration_s, transcription_time_s
        """
        # faster-whisper expects float32 normalized to [-1, 1]
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        duration_s = len(audio_f32) / sample_rate

        t0 = time.monotonic()

        kwargs = {}
        if self._language:
            kwargs["language"] = self._language

        segments, info = self._model.transcribe(audio_f32, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)

        elapsed = time.monotonic() - t0

        return {
            "text": text,
            "language": info.language,
            "duration_s": duration_s,
            "transcription_time_s": elapsed,
        }
