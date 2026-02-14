import importlib
import sys
import types

import numpy as np


def _load_audio_capture_with_fake_sounddevice(monkeypatch):
    fake_sd = types.SimpleNamespace(InputStream=object)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    sys.modules.pop("audio.capture", None)
    return importlib.import_module("audio.capture")


def test_dropped_frames_counter_increments_and_resets(monkeypatch) -> None:
    capture_mod = _load_audio_capture_with_fake_sounddevice(monkeypatch)
    capture = capture_mod.AudioCapture(
        {
            "sample_rate": 16000,
            "channels": 1,
            "blocksize": 1280,
            "ring_buffer_seconds": 1,
        }
    )

    indata = np.ones((1280, 1), dtype=np.float32)
    # Queue maxsize is 200, so this should force drops.
    for _ in range(260):
        capture._callback(indata, 1280, None, None)

    assert capture.dropped_frames > 0
    dropped = capture.consume_dropped_frames()
    assert dropped > 0
    assert capture.dropped_frames == 0
