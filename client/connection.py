"""WebSocket client with auto-reconnect and exponential backoff.

Runs the asyncio WebSocket receive loop in a background thread.
Incoming messages are pushed to a thread-safe queue for the main
(synchronous) frame loop to consume.
"""

import asyncio
import json
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

    def __init__(self, server_url: str, reconnect_min_s: float = 1.0, reconnect_max_s: float = 30.0):
        self._server_url = server_url
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s

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
        if self._loop and self._send_queue:
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, data)

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
