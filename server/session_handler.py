"""Per-connection pipeline orchestration for WebSocket sessions.

Dispatches incoming messages, runs the STT→LLM→TTS pipeline, and handles
barge-in cancellation via an asyncio.Event.
"""

import asyncio
import logging
import queue
import threading
import time
import traceback

import numpy as np
from fastapi import WebSocket

from shared import protocol
from server.stt.whisper_stt import WhisperSTT
from server.stt.filters import check_hallucination
from server.llm.openrouter_client import OpenRouterClient
from server.llm.prompt import get_system_prompt, build_messages, clean_for_tts
from server.tts import TTSEngine
from server.assistant.session import Session
from server.assistant.metrics import MetricsLogger
from server.assistant.language import detect_response_language
from server.assistant.telemetry import stt_metrics_payload, llm_metrics_payload

log = logging.getLogger(__name__)

_ERROR_MESSAGES: dict[str, str] = {
    "en": "Sorry, something went wrong.",
    "de": "Entschuldigung, da ist etwas schiefgelaufen.",
}

_PIPELINE_ERROR_CODES: dict[str, str] = {
    "stt": "pipeline_stt_failed",
    "llm": "pipeline_llm_failed",
    "tts": "pipeline_tts_failed",
}


class SessionHandler:
    """Manages one WebSocket connection's pipeline and state."""

    def __init__(
        self,
        ws: WebSocket,
        stt: WhisperSTT,
        llm: OpenRouterClient,
        tts: TTSEngine,
        session: Session,
        metrics: MetricsLogger,
        config: dict,
    ):
        self._ws = ws
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._session = session
        self._metrics = metrics
        self._config = config
        self._cancel_event = asyncio.Event()
        self._pipeline_task: asyncio.Task | None = None

        metrics_cfg = config.get("metrics", {})
        self._log_transcripts = metrics_cfg.get("log_transcripts", False)
        self._log_llm_text = metrics_cfg.get("log_llm_text", False)

        protocol_cfg = config.get("protocol", {})
        try:
            mismatch_reject_ratio = float(protocol_cfg.get("audio_mismatch_reject_ratio", 0.2))
        except (TypeError, ValueError):
            mismatch_reject_ratio = 0.2
        self._audio_mismatch_reject_ratio = min(max(mismatch_reject_ratio, 0.0), 1.0)

    async def handle(self) -> None:
        """Main receive loop — dispatches messages and runs pipeline."""
        try:
            while True:
                msg = await self._ws.receive()

                if msg["type"] == "websocket.disconnect":
                    break

                if msg["type"] == "websocket.receive":
                    if "text" in msg:
                        await self._handle_text(msg["text"])
                    # Binary frames are handled after receiving utterance_audio meta
        except Exception as e:
            log.warning("WebSocket connection error: %s", e)
        finally:
            # Wait for in-flight pipeline to complete before cleanup
            if self._pipeline_task and not self._pipeline_task.done():
                try:
                    await asyncio.wait_for(self._pipeline_task, timeout=30)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._cancel_event.set()
                    self._pipeline_task.cancel()
                except Exception:
                    pass
            self._metrics.flush()

    async def _handle_text(self, text: str) -> None:
        """Dispatch a JSON text message."""
        msg = protocol.decode_json(text)
        msg_type = msg.get("type")

        if msg_type == protocol.WAKE:
            await self._on_wake(msg)
        elif msg_type == protocol.UTTERANCE_AUDIO:
            await self._on_utterance_audio(msg)
        elif msg_type == protocol.BARGE_IN:
            self._on_barge_in()
        elif msg_type == protocol.FOLLOW_UP_TIMEOUT:
            await self._on_follow_up_timeout()
        else:
            log.warning("Unknown message type: %s", msg_type)

    async def _on_wake(self, msg: dict) -> None:
        """Handle wake word detection — start LLM warmup."""
        score = msg.get("score", 0.0)
        self._metrics.log("wake_detected", score=score)
        log.info("Wake word detected (score=%.2f), warming up LLM", score)

        # Warmup in thread pool (blocking HTTP call)
        asyncio.get_event_loop().run_in_executor(None, self._llm.warmup)
        await self._ws.send_text(protocol.make_warmup_ack())

    async def _on_utterance_audio(self, meta: dict) -> None:
        """Receive utterance audio (meta JSON + following binary frame)."""
        try:
            sample_rate = int(meta.get("sample_rate", 16000))
            expected_samples = int(meta.get("samples", 0))
        except (TypeError, ValueError):
            self._metrics.log("protocol_invalid_audio_meta")
            await self._ws.send_text(
                protocol.make_error(
                    "Invalid audio metadata.",
                    stage="protocol",
                    code="protocol_invalid_audio_meta",
                )
            )
            return

        if sample_rate <= 0 or expected_samples < 0:
            self._metrics.log("protocol_invalid_audio_meta", sample_rate=sample_rate, samples=expected_samples)
            await self._ws.send_text(
                protocol.make_error(
                    "Invalid audio metadata.",
                    stage="protocol",
                    code="protocol_invalid_audio_meta",
                )
            )
            return

        # Next frame should be the binary audio data
        binary_msg = await self._ws.receive()
        if binary_msg["type"] != "websocket.receive" or "bytes" not in binary_msg:
            await self._ws.send_text(
                protocol.make_error(
                    "Expected binary audio frame after utterance metadata.",
                    stage="protocol",
                    code="protocol_expected_audio_binary",
                )
            )
            return

        audio_bytes = binary_msg["bytes"]
        audio_int16 = protocol.decode_audio(audio_bytes)
        actual_samples = len(audio_int16)

        if actual_samples != expected_samples:
            mismatch_ratio = abs(actual_samples - expected_samples) / max(actual_samples, expected_samples, 1)
            payload = {
                "expected_samples": expected_samples,
                "actual_samples": actual_samples,
                "mismatch_ratio": round(mismatch_ratio, 4),
            }
            if mismatch_ratio > self._audio_mismatch_reject_ratio:
                log.warning(
                    "Rejecting audio payload mismatch: expected=%d actual=%d ratio=%.3f",
                    expected_samples, actual_samples, mismatch_ratio,
                )
                self._metrics.log("protocol_audio_mismatch_rejected", **payload)
                await self._ws.send_text(
                    protocol.make_error(
                        "Audio payload did not match metadata.",
                        stage="protocol",
                        code="protocol_audio_size_mismatch",
                    )
                )
                return

            log.warning(
                "Accepting minor audio payload mismatch: expected=%d actual=%d ratio=%.3f",
                expected_samples, actual_samples, mismatch_ratio,
            )
            self._metrics.log("protocol_audio_mismatch_small", **payload)

        log.info("Received utterance: %d samples (expected %d) at %d Hz",
                 actual_samples, expected_samples, sample_rate)

        # Cancel any existing pipeline
        if self._pipeline_task and not self._pipeline_task.done():
            self._cancel_event.set()
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

        # Run pipeline as a task so receive loop continues for barge_in
        self._cancel_event.clear()
        self._pipeline_task = asyncio.create_task(
            self._run_pipeline(audio_int16, sample_rate)
        )

    def _on_barge_in(self) -> None:
        """Handle barge-in — signal pipeline to stop TTS streaming."""
        log.info("Barge-in received")
        self._metrics.log("barge_in")
        self._cancel_event.set()

    async def _on_follow_up_timeout(self) -> None:
        """Handle follow-up timeout — clear session."""
        log.info("Follow-up timeout, clearing session")
        self._session.clear()
        await self._ws.send_text(protocol.make_session_cleared())

    async def _stream_tts_incremental(
        self,
        text: str,
        language: str,
    ) -> tuple[int, bool]:
        """Incrementally stream TTS chunks while allowing mid-stream cancel."""
        stream_queue: queue.Queue = queue.Queue()
        stream_done = object()
        stop_requested = threading.Event()

        def _producer() -> None:
            try:
                for chunk in self._tts.synthesize_chunks(text, language=language):
                    if stop_requested.is_set() or self._cancel_event.is_set():
                        break
                    stream_queue.put(chunk)
            except Exception as exc:
                stream_queue.put(exc)
            finally:
                stream_queue.put(stream_done)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        cancelled = False
        chunk_index = 0

        while True:
            if self._cancel_event.is_set():
                cancelled = True
                stop_requested.set()
                break

            try:
                item = await asyncio.to_thread(stream_queue.get, True, 0.1)
            except queue.Empty:
                continue
            if item is stream_done:
                break
            if isinstance(item, Exception):
                raise item

            audio_int16_chunk, sr, is_last = item
            if self._cancel_event.is_set():
                cancelled = True
                stop_requested.set()
                break

            if len(audio_int16_chunk) == 0:
                continue

            await self._ws.send_text(protocol.make_tts_audio_meta(
                sample_rate=sr,
                num_samples=len(audio_int16_chunk),
                chunk_index=chunk_index,
                is_last=is_last,
            ))
            await self._ws.send_bytes(protocol.encode_audio(audio_int16_chunk))
            chunk_index += 1

        stop_requested.set()
        await asyncio.to_thread(producer_thread.join, 0.05)
        if producer_thread.is_alive():
            log.warning("TTS producer thread still running after cancellation window")

        return chunk_index, cancelled

    async def _run_pipeline(self, audio_int16: np.ndarray, sample_rate: int) -> None:
        """Run the full STT → LLM → TTS pipeline."""
        interaction_start = time.monotonic()
        detected_lang: str | None = None
        pipeline_stage = "stt"

        try:
            # ── STT ─────────────────────────────────────────────
            await self._ws.send_text(protocol.make_status(protocol.STAGE_STT_START))

            stt_result = await asyncio.to_thread(
                self._stt.transcribe, audio_int16, sample_rate
            )
            transcript = stt_result["text"]
            detected_lang = stt_result.get("language")
            avg_logprob = stt_result["avg_logprob"]
            no_speech_prob = stt_result["no_speech_prob"]

            log.info("STT: '%s' (lang=%s, %.2fs, logprob=%.2f, no_speech=%.2f)",
                     transcript[:80], detected_lang,
                     stt_result["transcription_time_s"],
                     avg_logprob, no_speech_prob)

            self._metrics.log(
                "stt_complete",
                **stt_metrics_payload(stt_result, include_text=self._log_transcripts),
            )
            await self._ws.send_text(protocol.make_status(protocol.STAGE_STT_COMPLETE))

            # Empty transcript
            if not transcript.strip():
                log.info("Empty transcript, rejecting")
                await self._ws.send_text(protocol.make_stt_rejected("empty_transcript"))
                return

            # Hallucination check
            stt_conf = self._config.get("stt", {})
            rejected, reason = check_hallucination(
                transcript, no_speech_prob, avg_logprob,
                no_speech_threshold=stt_conf.get("no_speech_threshold", 0.6),
                logprob_threshold=stt_conf.get("logprob_threshold", -1.0),
            )
            if rejected:
                log.info("STT rejected: %s", reason)
                rejected_payload = {"reason": reason, "text_chars": len(transcript)}
                if self._log_transcripts:
                    rejected_payload["text"] = transcript
                self._metrics.log("stt_rejected", **rejected_payload)
                await self._ws.send_text(protocol.make_stt_rejected(reason))
                return

            # ── LLM ─────────────────────────────────────────────
            pipeline_stage = "llm"
            await self._ws.send_text(protocol.make_status(protocol.STAGE_LLM_START))

            self._session.add_user_message(transcript)
            system_prompt = get_system_prompt(detected_lang)
            messages = build_messages(
                system_prompt,
                self._session.get_messages()[:-1],
                transcript,
            )

            llm_result = await asyncio.to_thread(self._llm.chat, messages)
            raw_response_text = llm_result["text"]
            response_text = clean_for_tts(raw_response_text)
            llm_result = {**llm_result, "text": response_text}

            if response_text != raw_response_text:
                self._metrics.log(
                    "llm_response_sanitized",
                    raw_chars=len(raw_response_text),
                    clean_chars=len(response_text),
                    removed_chars=max(0, len(raw_response_text) - len(response_text)),
                )

            log.info("LLM: '%s' (ttft=%.2fs, total=%.2fs)",
                     response_text[:80], llm_result["ttft_s"], llm_result["elapsed_s"])

            self._metrics.log(
                "llm_complete",
                **llm_metrics_payload(llm_result, include_text=self._log_llm_text),
            )
            await self._ws.send_text(protocol.make_status(protocol.STAGE_LLM_COMPLETE))

            if not response_text.strip():
                log.info("Empty LLM response")
                await self._ws.send_text(protocol.make_tts_done(cancelled=False))
                return

            self._session.add_assistant_message(response_text)

            # Detect response language for TTS voice
            response_lang = detect_response_language(
                response_text, fallback=detected_lang or "en"
            )

            # ── TTS — stream per-sentence chunks ────────────────
            pipeline_stage = "tts"
            await self._ws.send_text(protocol.make_status(protocol.STAGE_TTS_START))

            tts_start = time.monotonic()
            chunk_index, cancelled = await self._stream_tts_incremental(
                response_text, language=response_lang
            )

            tts_elapsed = time.monotonic() - tts_start
            log.info("TTS: %d chunks in %.2fs (input=%s, voice=%s, cancelled=%s)",
                     chunk_index, tts_elapsed, detected_lang, response_lang, cancelled)

            self._metrics.log(
                "tts_complete",
                duration_s=tts_elapsed,
                input_language=detected_lang,
                voice_language=response_lang,
                chunks=chunk_index,
                cancelled=cancelled,
            )

            await self._ws.send_text(protocol.make_tts_done(cancelled=cancelled))

            # Log total pipeline latency
            total_elapsed = time.monotonic() - interaction_start
            self._metrics.log(
                "interaction_complete",
                total_elapsed_s=total_elapsed,
                stt_time_s=stt_result["transcription_time_s"],
                llm_ttft_s=llm_result["ttft_s"],
                llm_total_s=llm_result["elapsed_s"],
                tts_time_s=tts_elapsed,
                input_language=detected_lang,
                voice_language=response_lang,
            )

        except asyncio.CancelledError:
            log.info("Pipeline cancelled")
            raise
        except Exception as e:
            log.error("Pipeline error: %s\n%s", e, traceback.format_exc())
            error_code = _PIPELINE_ERROR_CODES.get(pipeline_stage, "pipeline_internal_error")
            self._metrics.log("pipeline_error", code=error_code, stage=pipeline_stage)
            self._metrics.log("pipeline_error_code", code=error_code, stage=pipeline_stage)
            try:
                err_lang = detected_lang or "en"
                err_msg = _ERROR_MESSAGES.get(err_lang, _ERROR_MESSAGES["en"])
                await self._ws.send_text(
                    protocol.make_error(
                        err_msg,
                        stage=pipeline_stage,
                        code=error_code,
                    )
                )
                # Try to speak error message
                for audio_chunk, sr, is_last in self._tts.synthesize_chunks(err_msg, language=err_lang):
                    if len(audio_chunk) == 0:
                        continue
                    await self._ws.send_text(protocol.make_tts_audio_meta(
                        sample_rate=sr,
                        num_samples=len(audio_chunk),
                        chunk_index=0,
                        is_last=is_last,
                    ))
                    await self._ws.send_bytes(protocol.encode_audio(audio_chunk))
                await self._ws.send_text(protocol.make_tts_done(cancelled=False))
            except Exception:
                pass
