"""Audio playback with instant stop for barge-in support."""

import threading
import numpy as np
import sounddevice as sd


class AudioPlayer:
    """Plays audio via sounddevice with support for instant interruption."""

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._playing = threading.Event()

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Start playback. Non-blocking — use wait_until_done() or is_playing."""
        sr = sample_rate or self._sample_rate
        self._playing.set()

        sd.play(audio, samplerate=sr)
        threading.Thread(target=self._monitor, args=(audio, sr), daemon=True).start()

    def _monitor(self, audio: np.ndarray, sample_rate: int) -> None:
        """Wait for playback to finish naturally, then clear the flag."""
        sd.wait()
        self._playing.clear()

    def stop(self) -> None:
        """Immediately stop playback (for barge-in)."""
        sd.stop()
        self._playing.clear()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until playback finishes. Returns True if finished, False on timeout."""
        if not self._playing.is_set():
            return True
        self._playing.wait(timeout=0)
        if not self._playing.is_set():
            return True
        start = __import__("time").monotonic()
        while self._playing.is_set():
            __import__("time").sleep(0.02)
            if timeout and (__import__("time").monotonic() - start) >= timeout:
                return False
        return True
