import asyncio
import importlib
import json
import sys
import types


def _load_connection_with_stubs(monkeypatch):
    fake_exceptions = types.ModuleType("websockets.exceptions")
    fake_exceptions.ConnectionClosed = Exception
    fake_exceptions.InvalidURI = Exception

    fake_websockets = types.ModuleType("websockets")
    fake_websockets.connect = lambda *args, **kwargs: None
    fake_websockets.exceptions = fake_exceptions

    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)
    monkeypatch.setitem(sys.modules, "websockets.exceptions", fake_exceptions)

    sys.modules.pop("client.connection", None)
    return importlib.import_module("client.connection")


class _InlineLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def test_offline_messages_are_buffered_and_flushed_in_order(monkeypatch):
    conn_mod = _load_connection_with_stubs(monkeypatch)
    conn = conn_mod.ServerConnection(
        "ws://example.test/ws",
        offline_send_buffer_size=10,
        offline_send_ttl_s=5.0,
    )

    conn.send_wake(0.9)
    conn.send_barge_in()

    assert len(conn._offline_send_buffer) == 2

    conn._send_queue = asyncio.Queue()
    conn._connected.set()
    conn._drain_offline_send_buffer()

    queued = [conn._send_queue.get_nowait(), conn._send_queue.get_nowait()]
    queued_types = [json.loads(item)["type"] for item in queued]
    assert queued_types == ["wake", "barge_in"]


def test_offline_buffer_drops_oldest_when_full(monkeypatch):
    conn_mod = _load_connection_with_stubs(monkeypatch)
    conn = conn_mod.ServerConnection(
        "ws://example.test/ws",
        offline_send_buffer_size=2,
        offline_send_ttl_s=5.0,
    )

    conn.send_wake(0.1)
    conn.send_barge_in()
    conn.send_follow_up_timeout()

    conn._send_queue = asyncio.Queue()
    conn._connected.set()
    conn._drain_offline_send_buffer()

    queued = [conn._send_queue.get_nowait(), conn._send_queue.get_nowait()]
    queued_types = [json.loads(item)["type"] for item in queued]
    assert queued_types == ["barge_in", "follow_up_timeout"]


def test_offline_buffer_drops_expired_messages(monkeypatch):
    conn_mod = _load_connection_with_stubs(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(conn_mod.time, "monotonic", lambda: now[0])

    conn = conn_mod.ServerConnection(
        "ws://example.test/ws",
        offline_send_buffer_size=10,
        offline_send_ttl_s=1.0,
    )
    conn.send_wake(0.5)

    now[0] = 102.0
    conn._send_queue = asyncio.Queue()
    conn._connected.set()
    conn._drain_offline_send_buffer()

    assert conn._send_queue.qsize() == 0


def test_connected_enqueue_writes_directly_to_send_queue(monkeypatch):
    conn_mod = _load_connection_with_stubs(monkeypatch)
    conn = conn_mod.ServerConnection("ws://example.test/ws")

    conn._loop = _InlineLoop()
    conn._send_queue = asyncio.Queue()
    conn._connected.set()

    conn.send_follow_up_timeout()

    queued = conn._send_queue.get_nowait()
    assert json.loads(queued)["type"] == "follow_up_timeout"
    assert len(conn._offline_send_buffer) == 0
