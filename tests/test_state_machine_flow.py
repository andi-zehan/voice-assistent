import importlib
import sys
import time
import types

import numpy as np


def _load_state_machine_with_stubs(monkeypatch):
    def _module(name: str, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    monkeypatch.setitem(sys.modules, "audio.capture", _module("audio.capture", AudioCapture=object))
    monkeypatch.setitem(sys.modules, "audio.playback", _module("audio.playback", AudioPlayer=object))
    monkeypatch.setitem(
        sys.modules,
        "audio.vad",
        _module("audio.vad", VoiceActivityDetector=object, UtteranceDetector=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "audio.earcon",
        _module("audio.earcon", play_earcon=lambda *a, **k: None, play_named_earcon=lambda *a, **k: None),
    )
    monkeypatch.setitem(sys.modules, "wake.detector", _module("wake.detector", WakeWordDetector=object))
    monkeypatch.setitem(sys.modules, "stt.whisper_stt", _module("stt.whisper_stt", WhisperSTT=object))
    monkeypatch.setitem(sys.modules, "llm.openrouter_client", _module("llm.openrouter_client", OpenRouterClient=object))
    monkeypatch.setitem(sys.modules, "tts", _module("tts", TTSEngine=object))

    sys.modules.pop("assistant.state_machine", None)
    return importlib.import_module("assistant.state_machine")


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
        self.wait_calls = 0
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
        self.wait_calls += 1
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
        self.process_calls = 0
        self.return_sequence = []

    def reset(self):
        self.reset_calls += 1
        self.state = "waiting"

    def process(self, frame, is_speech):
        self.process_calls += 1
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


class FakeSTT:
    def __init__(self, result=None, error=None):
        self._result = result or {
            "text": "hello",
            "language": "en",
            "duration_s": 0.5,
            "transcription_time_s": 0.1,
            "avg_logprob": -0.1,
            "no_speech_prob": 0.01,
        }
        self._error = error

    def transcribe(self, audio, sample_rate):
        if self._error:
            raise self._error
        return dict(self._result)


class FakeLLM:
    def __init__(self, result=None, error=None):
        self._result = result or {
            "text": "ok",
            "model": "fake",
            "elapsed_s": 0.2,
            "ttft_s": 0.1,
        }
        self._error = error
        self.warmup_calls = 0

    def warmup(self):
        self.warmup_calls += 1

    def chat(self, messages):
        if self._error:
            raise self._error
        return dict(self._result)


class FakeTTS:
    def __init__(self, error=None):
        self._error = error

    def synthesize(self, text, language=None):
        if self._error:
            raise self._error
        return np.array([0.0, 0.1], dtype=np.float32), 16000


class FakeSession:
    def __init__(self):
        self.history = []
        self.clear_calls = 0

    def add_user_message(self, text):
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text):
        self.history.append({"role": "assistant", "content": text})

    def get_messages(self):
        return list(self.history)

    def clear(self):
        self.clear_calls += 1
        self.history.clear()


class FakeMetrics:
    def __init__(self):
        self.events = []

    def log(self, event_type, **data):
        self.events.append((event_type, data))


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
        "stt": {},
        "conversation": {"follow_up_window_s": 3.0},
        "metrics": {"log_transcripts": True, "log_llm_text": True},
    }


def _build_machine(sm, **overrides):
    capture = overrides.get("capture", FakeCapture())
    player = overrides.get("player", FakePlayer())
    vad = overrides.get("vad", FakeVAD())
    utterance = overrides.get("utterance", FakeUtteranceDetector())
    wake = overrides.get("wake", FakeWakeDetector())
    stt = overrides.get("stt", FakeSTT())
    llm = overrides.get("llm", FakeLLM())
    tts = overrides.get("tts", FakeTTS())
    session = overrides.get("session", FakeSession())
    metrics = overrides.get("metrics", FakeMetrics())

    machine = sm.StateMachine(
        config=_base_config(),
        capture=capture,
        player=player,
        vad=vad,
        utterance_detector=utterance,
        wake_detector=wake,
        stt=stt,
        llm_client=llm,
        tts=tts,
        session=session,
        metrics=metrics,
    )
    return machine, capture, player, vad, utterance, wake, stt, llm, tts, session, metrics


def test_passive_to_listening_transition_on_wake(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    wake = FakeWakeDetector(detections=[(True, 0.9)])
    machine, _, _, _, utterance, wake, _, llm, _, _, metrics = _build_machine(sm, wake=wake)

    machine._handle_passive(np.zeros(1280, dtype=np.int16))

    assert machine.state == sm.State.LISTENING
    assert wake.reset_calls == 1
    assert utterance.reset_calls == 1
    assert llm.warmup_calls == 1
    assert any(name == "wake_detected" for name, _ in metrics.events)


def test_listening_soft_timeout_returns_passive(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _, _, session, metrics = _build_machine(sm)

    machine._state = sm.State.LISTENING
    machine._listening_start_time = 10.0
    monkeypatch.setattr(sm.time, "monotonic", lambda: 12.0)

    machine._handle_listening(np.zeros(1280, dtype=np.int16))

    assert machine.state == sm.State.PASSIVE
    assert session.clear_calls == 1
    assert any(name == "listening_timeout" for name, _ in metrics.events)


def test_speaking_barge_in_transitions_to_listening(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    vad = FakeVAD(speech_sequence=[True, True])
    machine, _, player, _, utterance, _, _, _, _, _, metrics = _build_machine(sm, vad=vad)

    machine._state = sm.State.SPEAKING
    player._playing = True
    machine._speaking_start_time = time.monotonic() - 2.0

    frame = np.zeros(1280, dtype=np.int16)
    machine._handle_speaking(frame)
    machine._handle_speaking(frame)

    assert player.stop_calls == 1
    assert utterance.reset_calls == 1
    assert machine.state == sm.State.LISTENING
    assert any(name == "barge_in" for name, _ in metrics.events)


def test_follow_up_timeout_returns_passive(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _, _, session, _ = _build_machine(sm)

    machine._state = sm.State.FOLLOW_UP
    machine._follow_up_deadline = 3.0
    monkeypatch.setattr(sm.time, "monotonic", lambda: 5.0)

    machine._check_follow_up_timeout()

    assert machine.state == sm.State.PASSIVE
    assert session.clear_calls == 1


def test_capture_drop_reporting_logs_metric(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    capture = FakeCapture()
    capture._drops = 7
    machine, _, _, _, _, _, _, _, _, _, metrics = _build_machine(sm, capture=capture)

    machine._report_capture_drops(100.0)

    assert any(name == "audio_frame_drop" and data.get("dropped_frames") == 7 for name, data in metrics.events)


def test_stt_failure_enters_follow_up(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _, _, _, metrics = _build_machine(sm, stt=FakeSTT(error=RuntimeError("stt failed")))

    machine._state = sm.State.THINKING
    machine._process_utterance(np.array([1, 2], dtype=np.int16))

    assert machine.state == sm.State.FOLLOW_UP
    assert any(name == "pipeline_error" for name, _ in metrics.events)


def test_llm_failure_enters_follow_up(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _, _, session, metrics = _build_machine(sm, llm=FakeLLM(error=RuntimeError("llm failed")))

    machine._state = sm.State.THINKING
    machine._process_utterance(np.array([1, 2], dtype=np.int16))

    assert machine.state == sm.State.FOLLOW_UP
    assert len(session.history) == 1
    assert any(name == "pipeline_error" for name, _ in metrics.events)


def test_tts_failure_enters_follow_up(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    machine, _, _, _, _, _, _, _, _, session, metrics = _build_machine(sm, tts=FakeTTS(error=RuntimeError("tts failed")))

    machine._state = sm.State.THINKING
    machine._process_utterance(np.array([1, 2], dtype=np.int16))

    assert machine.state == sm.State.FOLLOW_UP
    assert len(session.history) == 2
    assert any(name == "pipeline_error" for name, _ in metrics.events)


def test_llm_response_is_sanitized_before_session_and_metrics(monkeypatch):
    sm = _load_state_machine_with_stubs(monkeypatch)
    llm = FakeLLM(
        result={
            "text": "Das ist die Antwort【1†source】.\n\nQuellen:\n[1] https://example.com",
            "model": "fake",
            "elapsed_s": 0.2,
            "ttft_s": 0.1,
        }
    )
    machine, _, _, _, _, _, _, _, _, session, metrics = _build_machine(sm, llm=llm)

    machine._state = sm.State.THINKING
    machine._process_utterance(np.array([1, 2], dtype=np.int16))

    assert machine.state == sm.State.SPEAKING
    assert session.history[-1]["role"] == "assistant"
    assert "Quellen" not in session.history[-1]["content"]
    assert "https://" not in session.history[-1]["content"]

    llm_complete = [data for name, data in metrics.events if name == "llm_complete"]
    assert llm_complete
    assert "Quellen" not in llm_complete[-1].get("text", "")
    assert "https://" not in llm_complete[-1].get("text", "")
    assert any(name == "llm_response_sanitized" for name, _ in metrics.events)
