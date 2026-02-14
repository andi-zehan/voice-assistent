"""Tests for the client-side state machine."""

import importlib
import queue
import sys
import time
import types

import numpy as np


def _module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _load_client_sm_with_stubs(monkeypatch):
    """Load client state machine with stubbed hardware dependencies."""
    # Stub sounddevice (not available in CI)
    monkeypatch.setitem(sys.modules, "sounddevice", _module("sounddevice"))

    # Stub openwakeword
    fake_oww_model = type("Model", (), {"__init__": lambda *a, **kw: None})
    monkeypatch.setitem(
        sys.modules,
        "openwakeword.model",
        _module("openwakeword.model", Model=fake_oww_model),
    )
    monkeypatch.setitem(sys.modules, "openwakeword", _module("openwakeword"))

    # Stub webrtcvad
    monkeypatch.setitem(sys.modules, "webrtcvad", _module("webrtcvad"))

    # Stub websockets
    monkeypatch.setitem(sys.modules, "websockets", _module("websockets"))
    monkeypatch.setitem(
        sys.modules,
        "websockets.exceptions",
        _module("websockets.exceptions", ConnectionClosed=Exception, InvalidURI=Exception),
    )

    # Clear cached modules to force reimport
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("client."):
            sys.modules.pop(mod_name, None)

    return importlib.import_module("client.state_machine")


# ── Fakes ────────────────────────────────────────────────────────

class FakeCapture:
    def __init__(self):
        self._drops = 0

    def start(self):
        pass

    def get_frame(self, timeout=0.2):
        return None

    def consume_dropped_frames(self):
        drops = self._drops
        self._drops = 0
        return drops


class FakePlayer:
    def __init__(self):
        self._playing = False
        self.stop_calls = 0
        self.play_calls = 0

    def play(self, audio, sample_rate=None):
        self._playing = True
        self.play_calls += 1

    def stop(self):
        self._playing = False
        self.stop_calls += 1

    @property
    def is_playing(self):
        return self._playing

    def wait_until_done(self, timeout=None):
        self._playing = False
        return True


class FakeVAD:
    def __init__(self, speech_sequence=None, default=False):
        self._speech_sequence = list(speech_sequence or [])
        self._default = default

    def is_speech(self, frame):
        if self._speech_sequence:
            return self._speech_sequence.pop(0)
        return self._default


class FakeUtteranceDetector:
    def __init__(self):
        self.state = "waiting"
        self._audio = np.array([1, 2, 3], dtype=np.int16)
        self.reset_calls = 0
        self.return_sequence = []

    def reset(self):
        self.reset_calls += 1
        self.state = "waiting"

    def process(self, frame, is_speech):
        if self.return_sequence:
            self.state = self.return_sequence.pop(0)
            return self.state
        if is_speech:
            self.state = "collecting"
        return self.state

    def get_audio(self):
        return self._audio


class FakeWakeDetector:
    def __init__(self, detections=None):
        self._detections = list(detections or [(False, 0.0)])
        self.reset_calls = 0

    def process(self, frame):
        if self._detections:
            return self._detections.pop(0)
        return False, 0.0

    def reset(self):
        self.reset_calls += 1


class FakeConnection:
    def __init__(self):
        self.recv_queue = queue.Queue()
        self.sent_messages = []

    def send_wake(self, score):
        self.sent_messages.append(("wake", score))

    def send_utterance(self, audio, sample_rate):
        self.sent_messages.append(("utterance", len(audio), sample_rate))

    def send_barge_in(self):
        self.sent_messages.append(("barge_in",))

    def send_follow_up_timeout(self):
        self.sent_messages.append(("follow_up_timeout",))


class FakeChunkPlayer:
    def __init__(self):
        self._playing = False
        self.start_stream_calls = 0
        self.enqueue_calls = 0
        self.finish_stream_calls = 0
        self.cancel_calls = 0

    @property
    def is_playing(self):
        return self._playing

    def start_stream(self):
        self.start_stream_calls += 1
        self._playing = True

    def enqueue(self, audio, sample_rate):
        self.enqueue_calls += 1

    def finish_stream(self):
        self.finish_stream_calls += 1
        self._playing = False

    def cancel(self):
        self.cancel_calls += 1
        self._playing = False


def _base_config():
    return {
        "audio": {
            "sample_rate": 16000,
            "capture_drop_report_s": 5.0,
        },
        "vad": {
            "barge_in_enabled": True,
            "barge_in_frames": 2,
            "barge_in_grace_s": 0.0,
            "follow_up_grace_s": 0.0,
            "speech_onset_frames": 2,
            "listening_timeout_s": 1.0,
            "max_utterance_s": 10.0,
        },
        "earcon": {"volume": 0.3, "frequency": 880, "duration_s": 0.1},
        "conversation": {"follow_up_window_s": 3.0},
        "wake": {"model_name": "hey_jarvis", "threshold": 0.5},
    }


def _build_machine(sm, **overrides):
    capture = overrides.get("capture", FakeCapture())
    player = overrides.get("player", FakePlayer())
    vad = overrides.get("vad", FakeVAD())
    utterance = overrides.get("utterance", FakeUtteranceDetector())
    wake = overrides.get("wake", FakeWakeDetector())
    conn = overrides.get("connection", FakeConnection())
    chunk_player = overrides.get("chunk_player", FakeChunkPlayer())

    machine = sm.ClientStateMachine(
        config=_base_config(),
        capture=capture,
        player=player,
        vad=vad,
        utterance_detector=utterance,
        wake_detector=wake,
        connection=conn,
        chunk_player=chunk_player,
    )
    return machine, capture, player, vad, utterance, wake, conn, chunk_player


# ── Tests ────────────────────────────────────────────────────────

def test_passive_to_listening_on_wake(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    wake = FakeWakeDetector(detections=[(True, 0.9)])
    machine, _, _, _, utterance, wake, conn, _ = _build_machine(sm, wake=wake)

    machine._handle_passive(np.zeros(1280, dtype=np.int16))

    assert machine.state == sm.State.LISTENING
    assert wake.reset_calls == 1
    assert utterance.reset_calls == 1
    assert any(msg[0] == "wake" for msg in conn.sent_messages)


def test_listening_soft_timeout_returns_passive(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, conn, _ = _build_machine(sm)

    machine._state = sm.State.LISTENING
    machine._listening_start_time = 10.0
    monkeypatch.setattr(sm.time, "monotonic", lambda: 12.0)

    machine._handle_listening(np.zeros(1280, dtype=np.int16))

    assert machine.state == sm.State.PASSIVE
    assert any(msg[0] == "follow_up_timeout" for msg in conn.sent_messages)


def test_listening_utterance_complete_sends_to_server(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    vad = FakeVAD(default=True)
    utterance = FakeUtteranceDetector()
    utterance.return_sequence = ["complete"]
    machine, _, _, _, _, _, conn, _ = _build_machine(sm, vad=vad, utterance=utterance)

    machine._state = sm.State.LISTENING
    machine._listening_start_time = time.monotonic()
    machine._listening_hard_start = time.monotonic()

    machine._handle_listening(np.zeros(1280, dtype=np.int16))

    assert machine.state == sm.State.WAITING
    assert any(msg[0] == "utterance" for msg in conn.sent_messages)


def test_tts_audio_transitions_waiting_to_speaking(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, chunk_player = _build_machine(sm)

    machine._state = sm.State.WAITING

    meta = {"type": "tts_audio", "sample_rate": 22050, "chunk_index": 0, "is_last": False}
    audio = np.zeros(4410, dtype=np.int16)
    machine._on_tts_audio(meta, audio)

    assert machine.state == sm.State.SPEAKING
    assert chunk_player.start_stream_calls == 1
    assert chunk_player.enqueue_calls == 1


def test_tts_done_in_waiting_enters_follow_up(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _ = _build_machine(sm)

    machine._state = sm.State.WAITING
    machine._dispatch_server_message({"type": "tts_done", "cancelled": False})

    assert machine.state == sm.State.FOLLOW_UP


def test_stt_rejected_enters_follow_up(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _ = _build_machine(sm)

    machine._state = sm.State.WAITING
    machine._dispatch_server_message({"type": "stt_rejected", "reason": "hallucination"})

    assert machine.state == sm.State.FOLLOW_UP


def test_speaking_barge_in_transitions_to_listening(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    vad = FakeVAD(speech_sequence=[True, True])
    chunk_player = FakeChunkPlayer()
    chunk_player._playing = True
    machine, _, _, _, utterance, _, conn, _ = _build_machine(
        sm, vad=vad, chunk_player=chunk_player
    )

    machine._state = sm.State.SPEAKING
    machine._speaking_start_time = time.monotonic() - 2.0

    frame = np.zeros(1280, dtype=np.int16)
    machine._handle_speaking(frame)
    machine._handle_speaking(frame)

    assert chunk_player.cancel_calls == 1
    assert utterance.reset_calls == 1
    assert machine.state == sm.State.LISTENING
    assert any(msg[0] == "barge_in" for msg in conn.sent_messages)


def test_follow_up_timeout_returns_passive(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, conn, _ = _build_machine(sm)

    machine._state = sm.State.FOLLOW_UP
    machine._follow_up_deadline = 3.0
    monkeypatch.setattr(sm.time, "monotonic", lambda: 5.0)

    machine._check_follow_up_timeout()

    assert machine.state == sm.State.PASSIVE
    assert any(msg[0] == "follow_up_timeout" for msg in conn.sent_messages)


def test_follow_up_speech_transitions_to_listening(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    vad = FakeVAD(speech_sequence=[True, True])
    machine, _, _, _, utterance, _, _, _ = _build_machine(sm, vad=vad)

    machine._state = sm.State.FOLLOW_UP
    machine._follow_up_deadline = time.monotonic() + 10
    machine._follow_up_start_time = time.monotonic() - 1.0

    frame = np.zeros(1280, dtype=np.int16)
    machine._handle_follow_up(frame)
    machine._handle_follow_up(frame)

    assert machine.state == sm.State.LISTENING
    assert utterance.reset_calls == 1


def test_server_error_in_waiting_enters_follow_up(monkeypatch):
    sm = _load_client_sm_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _ = _build_machine(sm)

    machine._state = sm.State.WAITING
    machine._dispatch_server_message({"type": "error", "message": "STT failed", "stage": "stt"})

    assert machine.state == sm.State.FOLLOW_UP
