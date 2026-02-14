"""WebSocket protocol constants and helpers for client-server communication.

Text frames carry JSON control messages. Binary frames carry raw PCM int16 audio.
"""

import json
import struct

import numpy as np

# ── Client → Server message types ───────────────────────────────────

WAKE = "wake"
UTTERANCE_AUDIO = "utterance_audio"
BARGE_IN = "barge_in"
FOLLOW_UP_TIMEOUT = "follow_up_timeout"

# ── Server → Client message types ───────────────────────────────────

WARMUP_ACK = "warmup_ack"
STATUS = "status"
STT_REJECTED = "stt_rejected"
TTS_AUDIO = "tts_audio"
TTS_DONE = "tts_done"
SESSION_CLEARED = "session_cleared"
ERROR = "error"

# ── Status stages ───────────────────────────────────────────────────

STAGE_STT_START = "stt_start"
STAGE_STT_COMPLETE = "stt_complete"
STAGE_LLM_START = "llm_start"
STAGE_LLM_COMPLETE = "llm_complete"
STAGE_TTS_START = "tts_start"


# ── Encoding helpers ────────────────────────────────────────────────

def encode_json(msg: dict) -> str:
    """Encode a control message as a JSON text frame."""
    return json.dumps(msg, separators=(",", ":"))


def decode_json(text: str) -> dict:
    """Decode a JSON text frame into a dict."""
    return json.loads(text)


def encode_audio(audio_int16: np.ndarray) -> bytes:
    """Encode an int16 numpy array as raw PCM bytes for a binary frame."""
    return audio_int16.tobytes()


def decode_audio(data: bytes) -> np.ndarray:
    """Decode raw PCM bytes from a binary frame into an int16 numpy array."""
    return np.frombuffer(data, dtype=np.int16).copy()


# ── Message constructors (client → server) ──────────────────────────

def make_wake(score: float) -> str:
    return encode_json({"type": WAKE, "score": round(score, 3)})


def make_utterance_meta(sample_rate: int, num_samples: int) -> str:
    return encode_json({
        "type": UTTERANCE_AUDIO,
        "sample_rate": sample_rate,
        "samples": num_samples,
    })


def make_barge_in() -> str:
    return encode_json({"type": BARGE_IN})


def make_follow_up_timeout() -> str:
    return encode_json({"type": FOLLOW_UP_TIMEOUT})


# ── Message constructors (server → client) ──────────────────────────

def make_warmup_ack() -> str:
    return encode_json({"type": WARMUP_ACK})


def make_status(stage: str) -> str:
    return encode_json({"type": STATUS, "stage": stage})


def make_stt_rejected(reason: str) -> str:
    return encode_json({"type": STT_REJECTED, "reason": reason})


def make_tts_audio_meta(
    sample_rate: int,
    num_samples: int,
    chunk_index: int,
    is_last: bool,
) -> str:
    return encode_json({
        "type": TTS_AUDIO,
        "sample_rate": sample_rate,
        "samples": num_samples,
        "chunk_index": chunk_index,
        "is_last": is_last,
    })


def make_tts_done(cancelled: bool = False) -> str:
    return encode_json({"type": TTS_DONE, "cancelled": cancelled})


def make_session_cleared() -> str:
    return encode_json({"type": SESSION_CLEARED})


def make_error(message: str, stage: str = "") -> str:
    msg = {"type": ERROR, "message": message}
    if stage:
        msg["stage"] = stage
    return encode_json(msg)
