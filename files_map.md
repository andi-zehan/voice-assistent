# Project File Map

Last updated: 2026-02-14

### prd_leonardo_voice_assistant_prototype_mac_all_in_one.md
Description: Product requirements document for the Leonardo voice assistant prototype. Defines the full pipeline, state machine, tech stack, milestones, and acceptance criteria.

### CLAUDE.md
Description: Claude Code guidance file with project overview, architecture, tech stack, and key design decisions.

### pyproject.toml
Description: Project metadata and dependencies for pip install -e . setup.

### config.yaml
Description: All runtime configuration — audio, wake word, VAD, STT, LLM, TTS, conversation, metrics, earcon.

### README.md
Description: Setup instructions, prerequisites, and usage guide.

### SECURITY.md
Description: Security and data-handling notes, including privacy defaults for metrics logging.

### main.py
Description: Entry point. Loads config, initializes all components, wires them into the state machine, handles shutdown.

### assistant/__init__.py
Description: Package init for assistant module.

### assistant/state_machine.py
Description: Central 5-state orchestrator (PASSIVE → LISTENING → THINKING → SPEAKING → FOLLOW_UP). Drives the full pipeline and handles barge-in.

### assistant/session.py
Description: Conversation history management with turn and token budget trimming.

### assistant/metrics.py
Description: Thread-safe JSONL event logger with buffered writes, validation, and non-fatal I/O failure handling.

### assistant/language.py
Description: Response-language detection helper for selecting the correct TTS voice with fallback support.

### assistant/telemetry.py
Description: Privacy-aware metrics payload builders for STT/LLM events.

### audio/__init__.py
Description: Package init for audio module.

### audio/ring_buffer.py
Description: Pre-allocated numpy circular buffer for continuous audio capture.

### audio/capture.py
Description: sounddevice.InputStream with callback, float32→int16 conversion, ring buffer + queue output.

### audio/vad.py
Description: WebRTC VAD wrapper (VoiceActivityDetector) and stateful UtteranceDetector for end-of-utterance detection.

### audio/playback.py
Description: sounddevice.play() wrapper with instant stop() for barge-in support.

### audio/earcon.py
Description: Sine-wave chime generation and playback for state transition notifications.

### wake/__init__.py
Description: Package init for wake word module.

### wake/detector.py
Description: openWakeWord wrapper for streaming wake word detection.

### stt/__init__.py
Description: Package init for STT module.

### stt/whisper_stt.py
Description: faster-whisper model loading and int16 audio transcription.

### llm/__init__.py
Description: Package init for LLM module.

### llm/openrouter_client.py
Description: OpenRouter streaming HTTP client with SSE parsing, warmup, web search support, and transient-failure retries.

### llm/prompt.py
Description: System prompt definition and message list builder.

### tts/__init__.py
Description: TTSEngine protocol definition and create_tts() factory for selecting Piper or macOS say backend.

### tts/mac_say.py
Description: macOS `say` → AIFF temp file → numpy array pipeline for TTS.

### tts/piper_tts.py
Description: Piper neural TTS backend — loads ONNX voice model and synthesizes float32 audio via PiperVoice.

### models/piper/.gitkeep
Description: Placeholder for Piper ONNX voice model files (gitignored).

### tests/test_metrics.py
Description: Tests for metrics flush interval coercion, write-failure tolerance, and serialization-failure handling.

### tests/test_state_machine_privacy.py
Description: Tests for privacy-safe STT/LLM telemetry payloads (raw text excluded by default).

### tests/test_language_detection.py
Description: Tests for EN/DE response language detection and fallback behavior.

### tests/test_openrouter_retries.py
Description: Tests for OpenRouter retry behavior on transient failures and no-retry behavior on 401 errors.

### tests/test_audio_capture_drops.py
Description: Tests for dropped-frame counters in audio capture under queue pressure.

### tests/files_map.md
Description: File map for the tests directory.
