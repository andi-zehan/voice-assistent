"""WebSocket client with auto-reconnect and exponential backoff.

Runs the asyncio WebSocket receive loop in a background thread.
Incoming messages are pushed to a thread-safe queue for the main
(synchronous) frame loop to consume.
"""

import asyncio
from collections import deque
import logging
import queue
import threading
import time

import numpy as np
import websockets
import websockets.exceptions

from shared import protocol

log = logging.getLogger(__name__)

# Sentinel for binary audio data following a tts_audio meta message
BINARY_AUDIO = "__binary_audio__"


class ServerConnection:
    """WebSocket client that runs in a background thread.

    Provides a thread-safe interface for the synchronous main loop:
    - ``send_*()`` methods enqueue outgoing messages
    - ``recv_queue`` provides incoming parsed messages
    """

    def __init__(
        self,
        server_url: str,
        reconnect_min_s: float = 1.0,
        reconnect_max_s: float = 30.0,
        offline_send_buffer_size: int = 200,
        offline_send_ttl_s: float = 5.0,
    ):
        self._server_url = server_url
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s

        try:
            parsed_buffer_size = int(offline_send_buffer_size)
        except (TypeError, ValueError):
            parsed_buffer_size = 200
        try:
            parsed_ttl_s = float(offline_send_ttl_s)
        except (TypeError, ValueError):
            parsed_ttl_s = 5.0

        self._offline_send_buffer_size = max(1, parsed_buffer_size)
        self._offline_send_ttl_s = max(0.1, parsed_ttl_s)
        self._offline_send_buffer: deque[tuple[float, str | bytes]] = deque()
        self._offline_send_lock = threading.Lock()

        # Incoming messages from server (parsed dicts or (meta_dict, audio_ndarray) tuples)
        self.recv_queue: queue.Queue = queue.Queue(maxsize=500)

        # Outgoing messages
        self._send_queue: asyncio.Queue | None = None

        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = threading.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        """Start the background WebSocket thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close connection."""
        self._running = False
        self._connected.clear()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def wait_connected(self, timeout: float = 30.0) -> bool:
        """Block until connected or timeout."""
        return self._connected.wait(timeout=timeout)

    # ── Outgoing message methods (thread-safe) ──────────────────

    def send_wake(self, score: float) -> None:
        self._enqueue_send(protocol.make_wake(score))

    def send_utterance(self, audio_int16: np.ndarray, sample_rate: int) -> None:
        """Send utterance meta + binary audio."""
        meta = protocol.make_utterance_meta(sample_rate, len(audio_int16))
        audio_bytes = protocol.encode_audio(audio_int16)
        self._enqueue_send(meta)
        self._enqueue_send(audio_bytes)

    def send_barge_in(self) -> None:
        self._enqueue_send(protocol.make_barge_in())

    def send_follow_up_timeout(self) -> None:
        self._enqueue_send(protocol.make_follow_up_timeout())

    def _enqueue_send(self, data: str | bytes) -> None:
        """Thread-safe enqueue for the async send loop."""
        if self._loop and self._send_queue and self._connected.is_set():
            try:
                self._loop.call_soon_threadsafe(self._send_queue.put_nowait, data)
                return
            except RuntimeError:
                # Event loop is shutting down; fall through to buffer.
                pass

        self._buffer_offline_send(data)

    def _drop_expired_offline_messages_locked(self, now_s: float) -> int:
        dropped = 0
        while self._offline_send_buffer:
            created_s = self._offline_send_buffer[0][0]
            if now_s - created_s <= self._offline_send_ttl_s:
                break
            self._offline_send_buffer.popleft()
            dropped += 1
        return dropped

    def _buffer_offline_send(self, data: str | bytes) -> None:
        now_s = time.monotonic()
        with self._offline_send_lock:
            dropped_expired = self._drop_expired_offline_messages_locked(now_s)
            if dropped_expired > 0:
                log.warning("Dropped %d expired outbound buffered messages", dropped_expired)

            if len(self._offline_send_buffer) >= self._offline_send_buffer_size:
                self._offline_send_buffer.popleft()
                log.warning("Offline send buffer full, dropping oldest outbound message")

            self._offline_send_buffer.append((now_s, data))

    def _drain_offline_send_buffer(self) -> None:
        if self._send_queue is None:
            return

        now_s = time.monotonic()
        buffered_data: list[str | bytes] = []

        with self._offline_send_lock:
            dropped_expired = self._drop_expired_offline_messages_locked(now_s)
            if dropped_expired > 0:
                log.warning("Dropped %d expired outbound buffered messages", dropped_expired)

            while self._offline_send_buffer:
                _, data = self._offline_send_buffer.popleft()
                buffered_data.append(data)

        for data in buffered_data:
            self._send_queue.put_nowait(data)

        if buffered_data:
            log.info("Flushed %d buffered outbound messages after reconnect", len(buffered_data))

    # ── Background asyncio loop ─────────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._send_queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            log.error("Connection loop exited: %s", e)
        finally:
            self._loop.close()

    async def _connect_loop(self) -> None:
        """Connect with exponential backoff, reconnect on failure."""
        backoff = self._reconnect_min_s

        while self._running:
            try:
                log.info("Connecting to %s ...", self._server_url)
                async with websockets.connect(self._server_url) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = self._reconnect_min_s
                    log.info("Connected to server")
                    self._drain_offline_send_buffer()

                    # Run send and receive concurrently
                    await asyncio.gather(
                        self._recv_loop(ws),
                        self._send_loop(ws),
                    )

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidURI,
                OSError,
                ConnectionRefusedError,
            ) as e:
                self._connected.clear()
                self._ws = None
                if not self._running:
                    break
                log.warning("Connection lost (%s), reconnecting in %.1fs...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max_s)

            except Exception as e:
                self._connected.clear()
                self._ws = None
                if not self._running:
                    break
                log.error("Unexpected connection error: %s, reconnecting in %.1fs...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max_s)

    async def _recv_loop(self, ws) -> None:
        """Receive messages and push to the thread-safe queue."""
        pending_meta: dict | None = None

        async for message in ws:
            if isinstance(message, str):
                parsed = protocol.decode_json(message)
                msg_type = parsed.get("type")

                if msg_type == protocol.TTS_AUDIO:
                    # Next binary frame is the audio data
                    pending_meta = parsed
                else:
                    try:
                        self.recv_queue.put_nowait(parsed)
                    except queue.Full:
                        log.warning("Recv queue full, dropping message: %s", msg_type)

            elif isinstance(message, bytes):
                if pending_meta is not None:
                    # Pair the binary audio with its meta
                    audio_int16 = protocol.decode_audio(message)
                    try:
                        self.recv_queue.put_nowait((pending_meta, audio_int16))
                    except queue.Full:
                        log.warning("Recv queue full, dropping TTS audio chunk")
                    pending_meta = None
                else:
                    log.warning("Received unexpected binary frame without meta")

    async def _send_loop(self, ws) -> None:
        """Drain the send queue and forward to WebSocket."""
        while True:
            data = await self._send_queue.get()
            try:
                if isinstance(data, bytes):
                    await ws.send(data)
                else:
                    await ws.send(data)
            except websockets.exceptions.ConnectionClosed:
                break
