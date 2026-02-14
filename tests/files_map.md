# Tests File Map

Last updated: 2026-02-14

### test_audio_capture_drops.py
Description: Verifies dropped-frame accounting under queue pressure and clipping behavior in float32->int16 capture conversion.

### test_chunk_player.py
Description: Verifies chunk queue playback order, cancellation, and empty-chunk handling for streamed TTS audio.

### test_client_state_machine.py
Description: Verifies client FSM transitions for wake/listen/wait/speak/follow-up, including timeout and barge-in behavior.

### test_connection.py
Description: Verifies client outbound buffering while disconnected, TTL expiration, drop-oldest policy, and reconnect flush ordering.

### test_integration.py
Description: End-to-end SessionHandler flow tests for wake-to-TTS pipeline, barge-in, session persistence, and protocol/error-code behavior.

### test_language_detection.py
Description: Verifies EN/DE response-language detection and fallback behavior for TTS voice selection.

### test_metrics.py
Description: Verifies metrics logger hardening for flush interval coercion, write failure tolerance, and serialization failures.

### test_openrouter_retries.py
Description: Verifies OpenRouter retry behavior for transient failures and fail-fast behavior on non-retryable 401 responses.

### test_prompt_cleaning.py
Description: Verifies removal of source/citation artifacts from LLM responses before TTS playback.

### test_protocol.py
Description: Verifies protocol constants and message/audio encode-decode helpers, including error code payload fields.

### test_session_handler.py
Description: Verifies server pipeline orchestration across warmup, metadata validation, incremental TTS streaming, cancellation, and sanitized error handling.
