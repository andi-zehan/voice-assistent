"""Voice Activity Detection using WebRTC VAD."""

import time
import numpy as np
import webrtcvad


class VoiceActivityDetector:
    """Wraps WebRTC VAD to classify audio frames as speech or silence.

    Handles splitting larger frames (e.g. 1280 samples / 80ms) into
    VAD-compatible sub-frames (320 samples / 20ms at 16kHz).

    An energy gate rejects quiet frames that WebRTC VAD might
    misclassify as speech (e.g. ambient noise, fan hum).
    """

    def __init__(self, vad_config: dict, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._frame_duration_ms = vad_config["frame_duration_ms"]
        self._frame_size = int(sample_rate * self._frame_duration_ms / 1000)
        self._energy_threshold = vad_config.get("energy_threshold", 300)

        self._vad = webrtcvad.Vad(vad_config["aggressiveness"])

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        """Check if any sub-frame in the given audio contains speech."""
        rms = np.sqrt(np.mean(frame_int16.astype(np.float32) ** 2))
        if rms < self._energy_threshold:
            return False

        audio_bytes = frame_int16.tobytes()
        chunk_bytes = self._frame_size * 2

        for offset in range(0, len(audio_bytes), chunk_bytes):
            chunk = audio_bytes[offset:offset + chunk_bytes]
            if len(chunk) < chunk_bytes:
                break
            if self._vad.is_speech(chunk, self._sample_rate):
                return True
        return False


class UtteranceDetector:
    """Stateful detector that tracks speech onset and end-of-utterance.

    States:
      - waiting: no speech detected yet
      - collecting: speech in progress, accumulating audio
      - complete: silence timeout reached, utterance is done
    """

    def __init__(self, vad_config: dict):
        self._silence_timeout_s = vad_config["silence_timeout_ms"] / 1000.0
        self._speech_onset_frames = vad_config["speech_onset_frames"]
        self._pre_buffer_size = self._speech_onset_frames + 4

        self._state = "waiting"
        self._consecutive_speech = 0
        self._last_speech_time = 0.0
        self._audio_chunks: list[np.ndarray] = []
        self._pre_buffer: list[np.ndarray] = []

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._state = "waiting"
        self._consecutive_speech = 0
        self._last_speech_time = 0.0
        self._audio_chunks.clear()
        self._pre_buffer.clear()

    def process(self, frame_int16: np.ndarray, is_speech: bool) -> str:
        """Feed a frame and return the current state."""
        now = time.monotonic()

        if self._state == "complete":
            return self._state

        if self._state == "waiting":
            self._pre_buffer.append(frame_int16.copy())
            if len(self._pre_buffer) > self._pre_buffer_size:
                self._pre_buffer.pop(0)

        if is_speech:
            self._consecutive_speech += 1
            self._last_speech_time = now

            if self._state == "waiting" and self._consecutive_speech >= self._speech_onset_frames:
                self._state = "collecting"
                self._audio_chunks.extend(self._pre_buffer)
                self._pre_buffer.clear()

            elif self._state == "collecting":
                self._audio_chunks.append(frame_int16.copy())
        else:
            self._consecutive_speech = 0

            if self._state == "collecting":
                self._audio_chunks.append(frame_int16.copy())
                if now - self._last_speech_time >= self._silence_timeout_s:
                    self._state = "complete"

        return self._state

    def get_audio(self) -> np.ndarray:
        """Return all collected audio as a single int16 array."""
        if not self._audio_chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._audio_chunks)
