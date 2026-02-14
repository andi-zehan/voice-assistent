"""Tests for the server-side session handler."""

import asyncio
import json
import sys
import types

import numpy as np
import pytest

from shared import protocol


# ── Fakes ────────────────────────────────────────────────────────

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
    def __init__(self, chunks=None, error=None):
        self._chunks = chunks
        self._error = error

    def synthesize_chunks(self, text, language=None):
        if self._error:
            raise self._error
        if self._chunks:
            yield from self._chunks
        else:
            audio = np.array([100, 200, 300], dtype=np.int16)
            yield audio, 22050, True


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

    def flush(self):
        pass


class FakeWebSocket:
    """Simulates a FastAPI WebSocket for testing."""

    def __init__(self, incoming_messages=None):
        self._incoming = list(incoming_messages or [])
        self._incoming_index = 0
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.client = "test-client"

    async def receive(self):
        if self._incoming_index < len(self._incoming):
            msg = self._incoming[self._incoming_index]
            self._incoming_index += 1
            return msg
        return {"type": "websocket.disconnect"}

    async def send_text(self, data: str):
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)


def _base_config():
    return {
        "stt": {
            "no_speech_threshold": 0.85,
            "logprob_threshold": -1.5,
        },
        "conversation": {
            "max_turns": 10,
            "max_tokens_budget": 8000,
        },
        "metrics": {
            "log_transcripts": True,
            "log_llm_text": True,
        },
    }


def _import_handler():
    """Import SessionHandler, stubbing heavy dependencies."""
    # Stub piper
    piper_voice_mod = types.ModuleType("piper.voice")
    piper_voice_mod.PiperVoice = type("PiperVoice", (), {})
    piper_config_mod = types.ModuleType("piper.config")
    piper_config_mod.SynthesisConfig = type("SynthesisConfig", (), {"__init__": lambda *a, **kw: None})
    piper_mod = types.ModuleType("piper")

    sys.modules.setdefault("piper", piper_mod)
    sys.modules.setdefault("piper.voice", piper_voice_mod)
    sys.modules.setdefault("piper.config", piper_config_mod)

    # Stub faster_whisper
    faster_whisper_mod = types.ModuleType("faster_whisper")
    faster_whisper_mod.WhisperModel = type("WhisperModel", (), {"__init__": lambda *a, **kw: None})
    sys.modules.setdefault("faster_whisper", faster_whisper_mod)

    # Clear cached server modules to pick up stubs
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("server."):
            sys.modules.pop(mod_name, None)

    from server.session_handler import SessionHandler
    return SessionHandler


# ── Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wake_triggers_warmup():
    Handler = _import_handler()

    llm = FakeLLM()
    ws = FakeWebSocket(incoming_messages=[
        {"type": "websocket.receive", "text": protocol.make_wake(0.85)},
    ])

    handler = Handler(
        ws=ws, stt=FakeSTT(), llm=llm, tts=FakeTTS(),
        session=FakeSession(), metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    # Should have sent warmup_ack
    ack_msgs = [m for m in ws.sent_text if '"warmup_ack"' in m]
    assert len(ack_msgs) == 1


@pytest.mark.asyncio
async def test_follow_up_timeout_clears_session():
    Handler = _import_handler()

    session = FakeSession()
    session.add_user_message("hello")

    ws = FakeWebSocket(incoming_messages=[
        {"type": "websocket.receive", "text": protocol.make_follow_up_timeout()},
    ])

    handler = Handler(
        ws=ws, stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS(),
        session=session, metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    assert session.clear_calls == 1
    # Wait for the async task to complete
    await asyncio.sleep(0.05)
    cleared_msgs = [m for m in ws.sent_text if '"session_cleared"' in m]
    assert len(cleared_msgs) == 1


@pytest.mark.asyncio
async def test_utterance_pipeline_sends_tts_chunks():
    Handler = _import_handler()

    audio = np.array([100, 200, 300], dtype=np.int16)
    audio_bytes = audio.tobytes()

    ws = FakeWebSocket(incoming_messages=[
        {
            "type": "websocket.receive",
            "text": protocol.make_utterance_meta(16000, len(audio)),
        },
        {"type": "websocket.receive", "bytes": audio_bytes},
    ])

    handler = Handler(
        ws=ws, stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS(),
        session=FakeSession(), metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    # Wait for pipeline task
    if handler._pipeline_task:
        await handler._pipeline_task

    # Should have sent status messages and TTS audio
    types_sent = [json.loads(m).get("type") for m in ws.sent_text]
    assert "status" in types_sent
    assert "tts_audio" in types_sent
    assert "tts_done" in types_sent
    assert len(ws.sent_bytes) >= 1


@pytest.mark.asyncio
async def test_stt_hallucination_sends_rejection():
    Handler = _import_handler()

    stt = FakeSTT(result={
        "text": "thank you for watching",
        "language": "en",
        "duration_s": 0.5,
        "transcription_time_s": 0.1,
        "avg_logprob": -0.1,
        "no_speech_prob": 0.01,
    })

    audio = np.array([100, 200], dtype=np.int16)
    ws = FakeWebSocket(incoming_messages=[
        {
            "type": "websocket.receive",
            "text": protocol.make_utterance_meta(16000, len(audio)),
        },
        {"type": "websocket.receive", "bytes": audio.tobytes()},
    ])

    handler = Handler(
        ws=ws, stt=stt, llm=FakeLLM(), tts=FakeTTS(),
        session=FakeSession(), metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    if handler._pipeline_task:
        await handler._pipeline_task

    types_sent = [json.loads(m).get("type") for m in ws.sent_text]
    assert "stt_rejected" in types_sent


@pytest.mark.asyncio
async def test_barge_in_cancels_tts():
    Handler = _import_handler()

    # Multi-chunk TTS that yields slowly
    chunks = [
        (np.array([100, 200], dtype=np.int16), 22050, False),
        (np.array([300, 400], dtype=np.int16), 22050, True),
    ]

    audio = np.array([100, 200], dtype=np.int16)
    ws = FakeWebSocket(incoming_messages=[
        {
            "type": "websocket.receive",
            "text": protocol.make_utterance_meta(16000, len(audio)),
        },
        {"type": "websocket.receive", "bytes": audio.tobytes()},
        # Barge-in arrives while pipeline is running
        {"type": "websocket.receive", "text": protocol.make_barge_in()},
    ])

    handler = Handler(
        ws=ws, stt=FakeSTT(), llm=FakeLLM(), tts=FakeTTS(chunks=chunks),
        session=FakeSession(), metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    if handler._pipeline_task:
        try:
            await handler._pipeline_task
        except asyncio.CancelledError:
            pass

    # Should have received the barge_in and logged it
    metrics = handler._metrics
    assert any(name == "barge_in" for name, _ in metrics.events)


@pytest.mark.asyncio
async def test_pipeline_error_sends_error_message():
    Handler = _import_handler()

    stt = FakeSTT(error=RuntimeError("model crashed"))

    audio = np.array([100, 200], dtype=np.int16)
    ws = FakeWebSocket(incoming_messages=[
        {
            "type": "websocket.receive",
            "text": protocol.make_utterance_meta(16000, len(audio)),
        },
        {"type": "websocket.receive", "bytes": audio.tobytes()},
    ])

    handler = Handler(
        ws=ws, stt=stt, llm=FakeLLM(), tts=FakeTTS(),
        session=FakeSession(), metrics=FakeMetrics(), config=_base_config(),
    )
    await handler.handle()

    if handler._pipeline_task:
        await handler._pipeline_task

    types_sent = [json.loads(m).get("type") for m in ws.sent_text]
    assert "error" in types_sent
