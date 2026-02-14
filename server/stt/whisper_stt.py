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
        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list)

        # Confidence metrics for hallucination filtering
        if seg_list:
            avg_logprob = sum(s.avg_logprob for s in seg_list) / len(seg_list)
            no_speech_prob = max(s.no_speech_prob for s in seg_list)
        else:
            avg_logprob = 0.0
            no_speech_prob = 1.0

        elapsed = time.monotonic() - t0

        return {
            "text": text,
            "language": info.language,
            "duration_s": duration_s,
            "transcription_time_s": elapsed,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
        }
