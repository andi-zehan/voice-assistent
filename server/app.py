"""FastAPI application with WebSocket endpoint for the voice assistant server."""

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.stt.whisper_stt import WhisperSTT
from server.llm.openrouter_client import OpenRouterClient
from server.tts import create_tts
from server.assistant.session import Session
from server.assistant.metrics import MetricsLogger
from server.session_handler import SessionHandler

log = logging.getLogger(__name__)

app = FastAPI(title="Leonardo Voice Assistant Server")

# Global references set by create_app()
_config: dict = {}
_stt: WhisperSTT | None = None
_llm: OpenRouterClient | None = None
_tts = None
_metrics: MetricsLogger | None = None


def create_app(config: dict) -> FastAPI:
    """Initialize server components and return the configured FastAPI app."""
    global _config, _stt, _llm, _tts, _metrics

    _config = config
    log.info("Initializing STT (model=%s, device=%s)...", config["stt"]["model_size"], config["stt"]["device"])
    _stt = WhisperSTT(config["stt"])

    log.info("Initializing LLM client (model=%s)...", config["llm"]["model"])
    _llm = OpenRouterClient(config["llm"])

    log.info("Initializing TTS...")
    _tts = create_tts(config["tts"])

    _metrics = MetricsLogger(config["metrics"])

    return app


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Single WebSocket endpoint — one concurrent connection at a time."""
    await ws.accept()
    log.info("Client connected: %s", ws.client)

    # Each connection gets its own session (conversation history)
    session = Session(_config["conversation"])

    handler = SessionHandler(
        ws=ws,
        stt=_stt,
        llm=_llm,
        tts=_tts,
        session=session,
        metrics=_metrics,
        config=_config,
    )

    try:
        await handler.handle()
    except WebSocketDisconnect:
        log.info("Client disconnected: %s", ws.client)
    finally:
        log.info("Connection closed: %s", ws.client)
