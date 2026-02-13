"""Thread-safe circular buffer backed by a pre-allocated numpy array."""

import threading
import numpy as np


class RingBuffer:
    """Fixed-size circular buffer for int16 audio samples.

    Pre-allocates a numpy array to avoid per-frame allocation.
    Thread-safe for single-writer / single-reader usage.
    """

    def __init__(self, max_seconds: float, sample_rate: int = 16000):
        self._capacity = int(max_seconds * sample_rate)
        self._buf = np.zeros(self._capacity, dtype=np.int16)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def write(self, data: np.ndarray) -> None:
        """Append samples to the buffer, wrapping around as needed."""
        n = len(data)
        with self._lock:
            if n >= self._capacity:
                # Data larger than buffer — keep only the tail
                self._buf[:] = data[-self._capacity:]
                self._write_pos = 0
                self._total_written += n
                return

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos:end] = data
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos:] = data[:first]
                self._buf[:n - first] = data[first:]

            self._write_pos = end % self._capacity
            self._total_written += n

    def read_last(self, num_samples: int) -> np.ndarray:
        """Return the most recent `num_samples` samples as a contiguous array."""
        with self._lock:
            available = min(num_samples, self._total_written, self._capacity)
            if available == 0:
                return np.zeros(0, dtype=np.int16)

            start = (self._write_pos - available) % self._capacity
            if start + available <= self._capacity:
                return self._buf[start:start + available].copy()
            else:
                first = self._capacity - start
                return np.concatenate([
                    self._buf[start:],
                    self._buf[:available - first],
                ])

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write_pos = 0
            self._total_written = 0
