"""Microphone capture using sounddevice with callback-based streaming."""

import queue
import threading
import numpy as np
import sounddevice as sd

from client.audio.ring_buffer import RingBuffer


class AudioCapture:
    """Captures audio from the default input device.

    The sounddevice callback converts float32 input to int16, writes to a ring
    buffer, and pushes frames onto a queue for the main loop to consume.
    """

    def __init__(self, audio_config: dict):
        self._sample_rate = audio_config["sample_rate"]
        self._channels = audio_config["channels"]
        self._blocksize = audio_config["blocksize"]
        self._dropped_frames = 0
        self._dropped_lock = threading.Lock()

        self.ring_buffer = RingBuffer(
            max_seconds=audio_config["ring_buffer_seconds"],
            sample_rate=self._sample_rate,
        )
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass  # Silently ignore xruns to avoid log spam
        mono = indata[:, 0] if indata.shape[1] > 1 else indata.ravel()
        mono_clipped = np.clip(mono, -1.0, 1.0)
        int16_data = (mono_clipped * 32767).astype(np.int16)
        self.ring_buffer.write(int16_data)
        try:
            self.frame_queue.put_nowait(int16_data)
        except queue.Full:
            with self._dropped_lock:
                self._dropped_frames += 1
            pass

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_frame(self, timeout: float = 0.2) -> np.ndarray | None:
        """Get the next audio frame from the queue. Returns None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def dropped_frames(self) -> int:
        with self._dropped_lock:
            return self._dropped_frames

    def consume_dropped_frames(self) -> int:
        """Return and reset dropped frame counter."""
        with self._dropped_lock:
            dropped = self._dropped_frames
            self._dropped_frames = 0
        return dropped
