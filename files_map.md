# Project File Map

Last updated: 2026-02-14

## Shared

### shared/__init__.py
Description: Package init for shared module.

### shared/protocol.py
Description: WebSocket protocol constants, message type enums, JSON/PCM encode/decode helpers, and message constructors for client-server communication.

## Client (Raspberry Pi)

### client/__init__.py
Description: Package init for client module.

### client/main.py
Description: Client entry point. Loads config, initializes audio/wake/VAD components, connects to server via WebSocket, runs the client state machine.

### client/state_machine.py
Description: Client-side 5-state FSM (PASSIVE -> LISTENING -> WAITING -> SPEAKING -> FOLLOW_UP). Handles wake word, VAD, earcons, barge-in. Delegates STT/LLM/TTS to server.

### client/connection.py
Description: WebSocket client with auto-reconnect (exponential backoff), bounded offline outbound buffering, TTL expiry, and reconnect flush.

### client/chunk_player.py
Description: TTS chunk queue and sequential playback. Receives int16 PCM chunks from server, plays via AudioPlayer with barge-in cancellation support.

### client/config.yaml
Description: Client configuration — server address, reconnect/buffering limits, audio device, wake word, VAD, earcon, and follow-up settings.

### client/audio/__init__.py
Description: Package init for client audio module.

### client/audio/capture.py
Description: sounddevice InputStream with callback, float32->int16 conversion, ring buffer + queue output.

### client/audio/ring_buffer.py
Description: Pre-allocated numpy circular buffer for continuous audio capture.

### client/audio/vad.py
Description: WebRTC VAD wrapper (VoiceActivityDetector) and stateful UtteranceDetector for end-of-utterance detection.

### client/audio/playback.py
Description: sounddevice play() wrapper with instant stop() for barge-in support.

### client/audio/earcon.py
Description: Sine-wave chime generation and playback for state transition notifications.

### client/wake/__init__.py
Description: Package init for client wake word module.

### client/wake/detector.py
Description: openWakeWord wrapper for streaming wake word detection.

## Server (Mini PC)

### server/__init__.py
Description: Package init for server module.

### server/main.py
Description: Server entry point. Loads config, initializes STT/LLM/TTS, starts uvicorn with FastAPI app.

### server/app.py
Description: FastAPI application with /ws WebSocket endpoint. Initializes server components and creates per-connection SessionHandler.

### server/session_handler.py
Description: Per-connection pipeline orchestration with protocol metadata validation, incremental TTS streaming, barge-in cancellation, and sanitized error responses.

### server/config.yaml
Description: Server configuration — bind address, STT model, LLM model/API, TTS voices, conversation limits, metrics settings, and protocol mismatch thresholds.

### server/stt/__init__.py
Description: Package init for server STT module.

### server/stt/whisper_stt.py
Description: faster-whisper model loading and int16 audio transcription.

### server/stt/filters.py
Description: STT hallucination detection — no_speech_prob, logprob threshold, and blocklist-based filtering.

### server/llm/__init__.py
Description: Package init for server LLM module.

### server/llm/openrouter_client.py
Description: OpenRouter streaming HTTP client with SSE parsing, warmup, web search support, and transient-failure retries.

### server/llm/prompt.py
Description: System prompt definition, response-cleaning utilities (citations/URLs/markup stripping), and message list builder.

### server/tts/__init__.py
Description: TTSEngine protocol definition and create_tts() factory (Piper only on server).

### server/tts/piper_tts.py
Description: Piper neural TTS backend — loads ONNX voice models, synthesize() for full audio, synthesize_chunks() generator for per-sentence streaming.

### server/assistant/__init__.py
Description: Package init for server assistant module.

### server/assistant/session.py
Description: Conversation history management with turn and token budget trimming.

### server/assistant/metrics.py
Description: Thread-safe JSONL event logger with buffered writes and non-fatal I/O failure handling.

### server/assistant/language.py
Description: Response-language detection helper for selecting the correct TTS voice.

### server/assistant/telemetry.py
Description: Privacy-aware metrics payload builders for STT/LLM events.

## Tests

### tests/test_protocol.py
Description: Tests for shared WebSocket protocol encode/decode, message constructors, and constants.

### tests/test_client_state_machine.py
Description: Client state machine tests — wake detection, listening timeouts, utterance sending, TTS playback, barge-in, follow-up transitions.

### tests/test_session_handler.py
Description: Server session handler tests — wake/warmup, protocol metadata validation, incremental TTS streaming/cancellation, and sanitized error handling.

### tests/test_chunk_player.py
Description: TTS chunk player tests — single/multi chunk playback, cancellation, empty chunk handling.

### tests/test_integration.py
Description: End-to-end integration tests — full pipeline wake-to-TTS, barge-in, session persistence, protocol mismatch rejection, and error code behavior.

### tests/test_prompt_cleaning.py
Description: Tests for citation/source stripping in assistant responses, including German source formats.

### tests/test_openrouter_retries.py
Description: Tests for OpenRouter retry behavior on transient failures and no-retry on 401.

### tests/test_audio_capture_drops.py
Description: Tests for dropped-frame counters and clipping-safe PCM conversion in audio capture.

### tests/test_connection.py
Description: Tests for client connection offline outbound buffering, TTL expiry, and reconnect flush ordering.

### tests/test_language_detection.py
Description: Tests for EN/DE response language detection and fallback behavior.

### tests/test_metrics.py
Description: Tests for metrics flush interval coercion, write-failure tolerance, and serialization-failure handling.

## Docs

### docs/robustness_checklist.md
Description: Manual reliability validation checklist covering core flows, protocol mismatch handling, reconnect buffering behavior, fault injection, and soak-test criteria.

### docs/deployment_agent_rpi4_ubuntu_minipc.md
Description: Agent-executable deployment specification for fresh-machine provisioning of Raspberry Pi 4 client and Ubuntu Mini PC server with systemd services.

## Configuration

### pyproject.toml
Description: Project metadata, dependencies (shared/client/server/dev extras), entry points, and pytest config.

### client_requirements.txt
Description: Lightweight client dependencies for Raspberry Pi (no ML models).

### server_requirements.txt
Description: Heavy server dependencies for Mini PC (faster-whisper, piper-tts, fastapi).

## Other

### scripts/soak_test.py
Description: Soak monitoring utility for long-run robustness checks.

### scripts/deploy_server.sh
Description: Agent-executable Ubuntu Mini PC server deployment script for provisioning dependencies, configuring runtime, and managing `leonardo-server` service.

### scripts/deploy_client.sh
Description: Agent-executable Raspberry Pi client deployment script for provisioning dependencies, configuring client runtime, and managing `leonardo-client` service.

### models/piper/.gitkeep
Description: Placeholder for Piper ONNX voice model files.

### CLAUDE.md
Description: Claude Code guidance file with project overview, architecture, and design decisions.

### README.md
Description: Setup instructions, prerequisites, and usage guide.

### SECURITY.md
Description: Security and data-handling notes.
