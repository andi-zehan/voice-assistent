"""TTS chunk queue and sequential playback.

Receives TTS audio chunks from the server and plays them sequentially
via the AudioPlayer. Supports barge-in (immediate stop + queue flush).
"""

import logging
import queue
import threading

import numpy as np

from client.audio.playback import AudioPlayer

log = logging.getLogger(__name__)


class ChunkPlayer:
    """Queues incoming TTS audio chunks and plays them sequentially.

    Thread-safe: chunks can be enqueued from any thread (typically the
    WebSocket receive thread), and playback runs on a dedicated thread.
    """

    def __init__(self, player: AudioPlayer):
        self._player = player
        self._chunk_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=100)
        self._thread: threading.Thread | None = None
        self._playing = False
        self._cancelled = False

    @property
    def is_playing(self) -> bool:
        """True while chunks are being played or queued."""
        return self._playing

    def start_stream(self) -> None:
        """Prepare for a new TTS stream. Call before enqueueing chunks."""
        self._cancelled = False
        self._playing = True
        # Drain any leftover chunks
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def enqueue(self, audio_int16: np.ndarray, sample_rate: int) -> None:
        """Add an audio chunk to the playback queue."""
        if self._cancelled:
            return
        # Convert int16 to float32 for sounddevice playback
        audio_f32 = audio_int16.astype(np.float32) / 32767.0
        try:
            self._chunk_queue.put_nowait((audio_f32, sample_rate))
        except queue.Full:
            log.warning("Chunk queue full, dropping TTS chunk")

    def finish_stream(self) -> None:
        """Signal that all chunks have been enqueued."""
        try:
            self._chunk_queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass

    def cancel(self) -> None:
        """Stop playback immediately (barge-in). Flushes the queue."""
        self._cancelled = True
        self._player.stop()
        # Drain queue
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break
        # Push sentinel to unblock playback thread
        try:
            self._chunk_queue.put_nowait(None)
        except queue.Full:
            pass

    def wait_done(self, timeout: float | None = None) -> bool:
        """Wait for all chunks to finish playing."""
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def _playback_loop(self) -> None:
        """Play chunks sequentially until sentinel or cancel."""
        try:
            while True:
                item = self._chunk_queue.get(timeout=10)
                if item is None:
                    break
                if self._cancelled:
                    break

                audio_f32, sample_rate = item
                if len(audio_f32) == 0:
                    continue

                self._player.play(audio_f32, sample_rate=sample_rate)
                self._player.wait_until_done(timeout=30)

                if self._cancelled:
                    break
        except queue.Empty:
            log.warning("Chunk playback timed out waiting for next chunk")
        finally:
            self._playing = False
