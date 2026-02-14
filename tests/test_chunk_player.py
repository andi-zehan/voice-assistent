"""Tests for the TTS chunk player."""

import time
import threading
import sys
import types

import numpy as np
import pytest


# Stub sounddevice before importing
_sd_mod = types.ModuleType("sounddevice")
_sd_mod.play = lambda *a, **kw: None
_sd_mod.wait = lambda: None
_sd_mod.stop = lambda: None
sys.modules.setdefault("sounddevice", _sd_mod)


class FakePlayer:
    """Simulates AudioPlayer for chunk player tests."""

    def __init__(self):
        self._playing = False
        self.played_chunks: list[tuple[np.ndarray, int]] = []

    def play(self, audio, sample_rate=None):
        self._playing = True
        self.played_chunks.append((audio, sample_rate))
        # Simulate short playback
        threading.Timer(0.01, self._finish).start()

    def _finish(self):
        self._playing = False

    def stop(self):
        self._playing = False

    @property
    def is_playing(self):
        return self._playing

    def wait_until_done(self, timeout=None):
        start = time.monotonic()
        while self._playing:
            time.sleep(0.005)
            if timeout and (time.monotonic() - start) >= timeout:
                return False
        return True


# Clear cached client modules
for mod in list(sys.modules.keys()):
    if mod.startswith("client."):
        sys.modules.pop(mod, None)

from client.chunk_player import ChunkPlayer


def test_single_chunk_plays():
    player = FakePlayer()
    cp = ChunkPlayer(player)

    cp.start_stream()
    chunk = np.array([100, 200, 300, 400], dtype=np.int16)
    cp.enqueue(chunk, 22050)
    cp.finish_stream()
    cp.wait_done(timeout=2)

    assert not cp.is_playing
    assert len(player.played_chunks) == 1
    assert player.played_chunks[0][1] == 22050


def test_multiple_chunks_play_sequentially():
    player = FakePlayer()
    cp = ChunkPlayer(player)

    cp.start_stream()
    for i in range(3):
        chunk = np.array([100 * (i + 1), 200 * (i + 1)], dtype=np.int16)
        cp.enqueue(chunk, 22050)
    cp.finish_stream()
    cp.wait_done(timeout=5)

    assert not cp.is_playing
    assert len(player.played_chunks) == 3


def test_cancel_stops_playback():
    player = FakePlayer()
    cp = ChunkPlayer(player)

    cp.start_stream()
    for i in range(10):
        chunk = np.array([100, 200], dtype=np.int16)
        cp.enqueue(chunk, 22050)

    time.sleep(0.02)  # Let playback start
    cp.cancel()
    cp.wait_done(timeout=2)

    assert not cp.is_playing
    # Should have played fewer than all 10 chunks
    assert len(player.played_chunks) < 10


def test_empty_chunk_skipped():
    player = FakePlayer()
    cp = ChunkPlayer(player)

    cp.start_stream()
    cp.enqueue(np.array([], dtype=np.int16), 22050)
    cp.enqueue(np.array([100, 200], dtype=np.int16), 22050)
    cp.finish_stream()
    cp.wait_done(timeout=2)

    assert len(player.played_chunks) == 1


def test_is_playing_reflects_state():
    player = FakePlayer()
    cp = ChunkPlayer(player)

    assert not cp.is_playing

    cp.start_stream()
    assert cp.is_playing

    cp.finish_stream()
    cp.wait_done(timeout=2)
    assert not cp.is_playing
