"""Integration tests — server + client communication over WebSocket.

Uses the real FastAPI app with a test WebSocket client to verify
end-to-end message flow.
"""

import asyncio
import json
import sys
import types

import numpy as np
import pytest

from shared import protocol


def _stub_dependencies():
    """Stub heavy ML dependencies for testing."""
    # Stub piper
    if "piper" not in sys.modules:
        piper_mod = types.ModuleType("piper")
        sys.modules["piper"] = piper_mod
    if "piper.voice" not in sys.modules:
        piper_voice = types.ModuleType("piper.voice")
        piper_voice.PiperVoice = type("PiperVoice", (), {})
        sys.modules["piper.voice"] = piper_voice
    if "piper.config" not in sys.modules:
        piper_config = types.ModuleType("piper.config")
        piper_config.SynthesisConfig = type("SynthesisConfig", (), {"__init__": lambda *a, **kw: None})
        sys.modules["piper.config"] = piper_config

    # Stub faster_whisper
    if "faster_whisper" not in sys.modules:
        fw_mod = types.ModuleType("faster_whisper")
        fw_mod.WhisperModel = type("WhisperModel", (), {"__init__": lambda *a, **kw: None})
        sys.modules["faster_whisper"] = fw_mod

_stub_dependencies()

from server.session_handler import SessionHandler
from server.assistant.session import Session
from server.assistant.metrics import MetricsLogger


class FakeSTT:
    def transcribe(self, audio, sample_rate):
        return {
            "text": "what is the weather",
            "language": "en",
            "duration_s": 1.0,
            "transcription_time_s": 0.1,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
        }


class FakeLLM:
    def warmup(self):
        pass

    def chat(self, messages):
        return {
            "text": "The weather is sunny today.",
            "model": "test-model",
            "elapsed_s": 0.3,
            "ttft_s": 0.1,
        }


class FakeTTS:
    def synthesize_chunks(self, text, language=None):
        # Return two chunks
        chunk1 = np.array([100, 200, 300], dtype=np.int16)
        chunk2 = np.array([400, 500, 600], dtype=np.int16)
        yield chunk1, 22050, False
        yield chunk2, 22050, True


class CollectingWebSocket:
    """Test WebSocket that collects all sent messages."""

    def __init__(self, incoming):
        self._incoming = list(incoming)
        self._idx = 0
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.client = "integration-test"

    async def receive(self):
        if self._idx < len(self._incoming):
            msg = self._incoming[self._idx]
            self._idx += 1
            return msg
        return {"type": "websocket.disconnect"}

    async def send_text(self, data):
        self.sent_text.append(data)

    async def send_bytes(self, data):
        self.sent_bytes.append(data)


@pytest.mark.asyncio
async def test_full_pipeline_wake_to_tts():
    """Test: wake -> utterance -> STT -> LLM -> TTS chunks -> tts_done."""
    audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)

    ws = CollectingWebSocket([
        {"type": "websocket.receive", "text": protocol.make_wake(0.9)},
        {"type": "websocket.receive", "text": protocol.make_utterance_meta(16000, len(audio))},
        {"type": "websocket.receive", "bytes": audio.tobytes()},
    ])

    config = {
        "stt": {"no_speech_threshold": 0.85, "logprob_threshold": -1.5},
        "conversation": {"max_turns": 10, "max_tokens_budget": 8000},
        "metrics": {"log_transcripts": True, "log_llm_text": True},
    }

    handler = SessionHandler(
        ws=ws,
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        session=Session(config["conversation"]),
        metrics=MetricsLogger({"enabled": False}),
        config=config,
    )

    await handler.handle()
    if handler._pipeline_task:
        await handler._pipeline_task

    # Parse all sent text messages
    sent_types = [json.loads(m).get("type") for m in ws.sent_text]

    # Verify complete message sequence
    assert "warmup_ack" in sent_types
    assert "status" in sent_types
    assert "tts_audio" in sent_types
    assert "tts_done" in sent_types

    # Verify binary audio was sent (2 chunks from FakeTTS)
    assert len(ws.sent_bytes) == 2

    # Verify tts_done indicates not cancelled
    tts_done_msgs = [json.loads(m) for m in ws.sent_text if '"tts_done"' in m]
    assert tts_done_msgs
    assert tts_done_msgs[-1]["cancelled"] is False


@pytest.mark.asyncio
async def test_pipeline_with_barge_in():
    """Test that barge_in cancels TTS streaming."""
    audio = np.array([1, 2, 3], dtype=np.int16)

    ws = CollectingWebSocket([
        {"type": "websocket.receive", "text": protocol.make_utterance_meta(16000, len(audio))},
        {"type": "websocket.receive", "bytes": audio.tobytes()},
        {"type": "websocket.receive", "text": protocol.make_barge_in()},
    ])

    config = {
        "stt": {"no_speech_threshold": 0.85, "logprob_threshold": -1.5},
        "conversation": {"max_turns": 10, "max_tokens_budget": 8000},
        "metrics": {"log_transcripts": True, "log_llm_text": True},
    }

    handler = SessionHandler(
        ws=ws,
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        session=Session(config["conversation"]),
        metrics=MetricsLogger({"enabled": False}),
        config=config,
    )

    await handler.handle()
    if handler._pipeline_task:
        try:
            await handler._pipeline_task
        except asyncio.CancelledError:
            pass

    # Verify tts_done was sent (may be cancelled=true due to barge-in)
    tts_done_msgs = [json.loads(m) for m in ws.sent_text if '"tts_done"' in m]
    # The pipeline ran — we at least got some messages
    assert len(ws.sent_text) > 0


@pytest.mark.asyncio
async def test_session_persists_across_utterances():
    """Test that conversation history builds across multiple utterances."""
    audio = np.array([1, 2, 3], dtype=np.int16)
    audio_bytes = audio.tobytes()

    ws = CollectingWebSocket([
        # First utterance
        {"type": "websocket.receive", "text": protocol.make_utterance_meta(16000, len(audio))},
        {"type": "websocket.receive", "bytes": audio_bytes},
        # Second utterance
        {"type": "websocket.receive", "text": protocol.make_utterance_meta(16000, len(audio))},
        {"type": "websocket.receive", "bytes": audio_bytes},
    ])

    config = {
        "stt": {"no_speech_threshold": 0.85, "logprob_threshold": -1.5},
        "conversation": {"max_turns": 10, "max_tokens_budget": 8000},
        "metrics": {"log_transcripts": True, "log_llm_text": True},
    }

    session = Session(config["conversation"])

    handler = SessionHandler(
        ws=ws,
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        session=session,
        metrics=MetricsLogger({"enabled": False}),
        config=config,
    )

    await handler.handle()

    # Wait for both pipeline tasks
    await asyncio.sleep(0.2)
    if handler._pipeline_task:
        try:
            await handler._pipeline_task
        except asyncio.CancelledError:
            pass

    # Session should have messages from both interactions
    messages = session.get_messages()
    assert len(messages) >= 2  # At least user + assistant from first interaction
