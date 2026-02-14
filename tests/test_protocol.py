"""Tests for the shared WebSocket protocol module."""

import json

import numpy as np

from shared import protocol


# ── Encoding / decoding ─────────────────────────────────────────

def test_encode_decode_json_roundtrip():
    msg = {"type": "wake", "score": 0.82}
    encoded = protocol.encode_json(msg)
    decoded = protocol.decode_json(encoded)
    assert decoded["type"] == "wake"
    assert decoded["score"] == 0.82


def test_encode_decode_audio_roundtrip():
    original = np.array([100, -200, 300, -400], dtype=np.int16)
    encoded = protocol.encode_audio(original)
    decoded = protocol.decode_audio(encoded)
    np.testing.assert_array_equal(original, decoded)


def test_encode_audio_empty():
    empty = np.array([], dtype=np.int16)
    encoded = protocol.encode_audio(empty)
    decoded = protocol.decode_audio(encoded)
    assert len(decoded) == 0


# ── Client -> Server constructors ────────────────────────────────

def test_make_wake():
    msg = json.loads(protocol.make_wake(0.82))
    assert msg["type"] == "wake"
    assert msg["score"] == 0.82


def test_make_utterance_meta():
    msg = json.loads(protocol.make_utterance_meta(16000, 32000))
    assert msg["type"] == "utterance_audio"
    assert msg["sample_rate"] == 16000
    assert msg["samples"] == 32000


def test_make_barge_in():
    msg = json.loads(protocol.make_barge_in())
    assert msg["type"] == "barge_in"


def test_make_follow_up_timeout():
    msg = json.loads(protocol.make_follow_up_timeout())
    assert msg["type"] == "follow_up_timeout"


# ── Server -> Client constructors ────────────────────────────────

def test_make_warmup_ack():
    msg = json.loads(protocol.make_warmup_ack())
    assert msg["type"] == "warmup_ack"


def test_make_status():
    msg = json.loads(protocol.make_status("stt_start"))
    assert msg["type"] == "status"
    assert msg["stage"] == "stt_start"


def test_make_stt_rejected():
    msg = json.loads(protocol.make_stt_rejected("hallucination_blocklist"))
    assert msg["type"] == "stt_rejected"
    assert msg["reason"] == "hallucination_blocklist"


def test_make_tts_audio_meta():
    msg = json.loads(protocol.make_tts_audio_meta(22050, 44100, 0, False))
    assert msg["type"] == "tts_audio"
    assert msg["sample_rate"] == 22050
    assert msg["samples"] == 44100
    assert msg["chunk_index"] == 0
    assert msg["is_last"] is False


def test_make_tts_done():
    msg = json.loads(protocol.make_tts_done(cancelled=True))
    assert msg["type"] == "tts_done"
    assert msg["cancelled"] is True


def test_make_tts_done_default():
    msg = json.loads(protocol.make_tts_done())
    assert msg["cancelled"] is False


def test_make_session_cleared():
    msg = json.loads(protocol.make_session_cleared())
    assert msg["type"] == "session_cleared"


def test_make_error():
    msg = json.loads(protocol.make_error("STT failed", stage="stt", code="pipeline_stt_failed"))
    assert msg["type"] == "error"
    assert msg["message"] == "STT failed"
    assert msg["stage"] == "stt"
    assert msg["code"] == "pipeline_stt_failed"


def test_make_error_no_stage():
    msg = json.loads(protocol.make_error("unknown error"))
    assert msg["type"] == "error"
    assert "stage" not in msg


def test_make_error_with_code_only():
    msg = json.loads(protocol.make_error("unknown error", code="internal_error"))
    assert msg["type"] == "error"
    assert msg["code"] == "internal_error"
    assert "stage" not in msg


# ── Constants ────────────────────────────────────────────────────

def test_message_type_constants():
    assert protocol.WAKE == "wake"
    assert protocol.UTTERANCE_AUDIO == "utterance_audio"
    assert protocol.BARGE_IN == "barge_in"
    assert protocol.FOLLOW_UP_TIMEOUT == "follow_up_timeout"
    assert protocol.WARMUP_ACK == "warmup_ack"
    assert protocol.STATUS == "status"
    assert protocol.STT_REJECTED == "stt_rejected"
    assert protocol.TTS_AUDIO == "tts_audio"
    assert protocol.TTS_DONE == "tts_done"
    assert protocol.SESSION_CLEARED == "session_cleared"
    assert protocol.ERROR == "error"


def test_stage_constants():
    assert protocol.STAGE_STT_START == "stt_start"
    assert protocol.STAGE_STT_COMPLETE == "stt_complete"
    assert protocol.STAGE_LLM_START == "llm_start"
    assert protocol.STAGE_LLM_COMPLETE == "llm_complete"
    assert protocol.STAGE_TTS_START == "tts_start"
