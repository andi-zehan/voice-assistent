# Tests File Map

Last updated: 2026-02-14

### test_metrics.py
Description: Verifies metrics logger hardening (flush interval validation, non-fatal write failures, serialization failure handling).

### test_state_machine_privacy.py
Description: Verifies privacy-aware telemetry payloads for STT and LLM events.

### test_language_detection.py
Description: Verifies EN/DE response-language detection and fallback behavior.

### test_openrouter_retries.py
Description: Verifies retry behavior for transient LLM API failures and fail-fast behavior on 401.

### test_audio_capture_drops.py
Description: Verifies capture dropped-frame counter increment/reset behavior when queue is full.

### test_state_machine_flow.py
Description: Simulates state-machine transitions and fault-injection recovery for STT/LLM/TTS failures.

### test_prompt_cleaning.py
Description: Verifies removal of citation/source artifacts (including German formats) from model responses.
