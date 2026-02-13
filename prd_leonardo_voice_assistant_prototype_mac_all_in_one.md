# PRD — Leonardo Voice Assistant Prototype (Mac, all-in-one)

## 1. Goal

Build a desktop prototype (macOS) of an Alexa-like voice assistant triggered by the wake word **“Leonardo”**.

Pipeline:

1. Wake word → 2) Capture user speech → 3) End-of-utterance detection → 4) STT via open-source Whisper → 5) LLM via OpenRouter (e.g., Gemini Flash) with **web search enabled via OpenRouter** → 6) TTS → 7) Playback → 8) Follow-up window for replies (or return to passive mode).

This prototype runs **entirely on a Mac** (no Raspberry Pi/satellite yet).

## 2. Non-Goals

- No custom hardware enclosure.
- No multi-room audio.
- No on-device (offline) LLM.
- No advanced personalization, long-term memory, or account integration.
- No mobile app.

## 3. Success Criteria

- Wake word reliability: triggers when “Leonardo” is spoken (acceptable false positive rate for a prototype).
- Latency: perceived responsiveness suitable for conversational use.
  - Target: first audible response starts within \~1.5–3.0 seconds after user stops speaking (best-effort; measure and iterate).
- Barge-in: user speech during TTS stops playback and starts listening again.
- Web-search grounding: assistant can answer questions requiring up-to-date info using OpenRouter’s online/web-search capability.
- Stable conversation loop: follow-up questions within a short window without repeating wake word.

## 4. Key User Stories

1. **Wake and ask**: As a user, I say “Leonardo” and ask a question; I get a spoken response.
2. **Follow-up**: After the response, I can ask a follow-up question without repeating “Leonardo” within the follow-up window.
3. **Barge-in**: While it’s speaking, I interrupt; it stops speaking and listens immediately.
4. **Web question**: I ask for recent information; it uses online search and cites sources verbally (“According to …”).
5. **Fail gracefully**: If STT/LLM fails, it asks to repeat or reports the failure succinctly.

## 5. Functional Requirements

### 5.1 Audio Capture

- Continuous microphone capture.
- Configurable input device selection.
- 16 kHz mono internal processing format.

### 5.2 Wake Word (“Leonardo”)

- Local wake word detection.
- On wake:
  - Play a short earcon (optional but recommended).
  - Start/mark a new interaction session.
  - Trigger LLM warm-up request (see 5.5) in parallel.

### 5.3 Voice Activity Detection (VAD) and End-of-Utterance

- Detect speech start and speech end.
- Use silence threshold + timeout (configurable; e.g., 500–900 ms).
- Collect audio segment for STT.

### 5.4 Speech-to-Text (STT)

- Use open-source Whisper implementation.
  - Preference: **whisper.cpp** (fast, local) or **faster-whisper** (GPU/Metal optional, depending on environment).
- Return transcript + confidence (if available) + timing.

### 5.5 LLM via OpenRouter

- Model configurable (default: a low-latency model such as Gemini Flash via OpenRouter).
- Streaming response enabled.
- **Warm-up on wake**:
  - When wake word fires, send a minimal streaming request (1–3 output tokens) and cancel/ignore output.
  - Purpose: reduce time-to-first-token for the real request.
- **Web search enabled via OpenRouter** (no server-side search tooling).
  - Use either `:online` variant or OpenRouter web-search plugin depending on supported model.
- Prompting:
  - System prompt optimized for spoken answers: concise, conversational, avoid long lists unless asked.
  - Instruct model to mention sources briefly when web search is used.

### 5.6 Text-to-Speech (TTS)

- Local TTS engine on Mac:
  - Phase 1: macOS `say` (fastest integration).
  - Phase 2: optionally Piper for consistent voices.
- Chunking:
  - Phase 1: speak after full LLM response.
  - Phase 2: stream tokens → sentence chunking → start speaking early.

### 5.7 Playback + Barge-in

- Playback audio while simultaneously running VAD.
- If user speech detected during playback:
  - Stop playback immediately.
  - Transition to listening state and capture new utterance.

### 5.8 Conversation State

- Maintain short conversation context (configurable max turns / token budget).
- Follow-up window after speaking (e.g., 3–6 seconds): if user speaks, treat it as continuation without wake word.
- If no speech in window, return to passive wake-word mode.

### 5.9 Observability

- Log timestamps for:
  - wake detected
  - speech start
  - speech end
  - STT start/end
  - LLM request start / first token / end
  - TTS start/end
  - barge-in events
- Store last N interactions’ transcripts + latencies.
- Optional: store audio clips for debugging (off by default).

## 6. Non-Functional Requirements

- Runs on macOS (Apple Silicon preferred but should also work on Intel if possible).
- Should run for hours without memory leaks.
- Privacy baseline: audio is processed locally except the LLM call; no continuous audio is sent to OpenRouter.

## 7. UX / Interaction Design

### 7.1 States

- **PASSIVE**: wake word running.
- **LISTENING**: capturing speech after wake.
- **THINKING**: STT + LLM.
- **SPEAKING**: playing TTS.
- **FOLLOW\_UP**: short window to continue without wake word.

### 7.2 Minimal UI

- Menu bar app or terminal UI (prototype):
  - Current state indicator.
  - Last transcript.
  - Latency metrics.
  - Toggle debug logging.

## 8. Technical Approach (Proposed)

### 8.1 Language/Runtime

- Preferred: Python (fast iteration) or Node.js.
- Recommendation: **Python** for audio + whisper + quick prototyping.

### 8.2 Components

- Audio I/O: `sounddevice` or `pyaudio`.
- VAD: WebRTC VAD.
- Wake word: `openWakeWord`.
- STT: `whisper.cpp` via CLI bindings or `faster-whisper`.
- LLM: OpenRouter HTTP API (streaming).
- TTS:
  - Phase 1: macOS `say`.
  - Phase 2: Piper.

### 8.3 Data Flow

- Ring buffer for mic audio.
- Wake triggers start marker.
- VAD segments utterance → WAV/PCM buffer.
- STT returns text.
- LLM request (with web search enabled) → response text.
- TTS converts to audio → playback.
- During playback, VAD monitors mic → barge-in.

## 9. Configuration

- Input device name/id
- Output device name/id
- Wake word threshold
- VAD aggressiveness + silence timeout
- Whisper model size
- OpenRouter:
  - API key
  - model slug
  - web-search mode (plugin vs `:online`)
  - max\_tokens
  - temperature
- Follow-up window duration

## 10. Decisions (v1 Prototype)

- Wake word: **Generic wake word detector is acceptable for v1**; defer custom “Leonardo” model training.
- TTS: **Use macOS `say` in v1**.
- Sentence-level speaking: **Defer to v2** (v1 speaks after full LLM response).
- Web search mode: **Use the simplest supported OpenRouter online mode**:
  - Prefer `:online` for the chosen model if available.
  - Otherwise enable OpenRouter’s web-search plugin.

## 11. Milestones

### M1 — End-to-end happy path (1 session)

- Wake → VAD segmentation → Whisper STT → OpenRouter LLM → TTS → playback.

### M2 — Conversation loop

- Follow-up window and context handling.

### M3 — Barge-in

- Stop playback on user speech; resume listening.

### M4 — Latency improvements

- Warm-up on wake.
- Streaming response.
- Optional sentence chunking.

### M5 — Web-search grounding

- Enable OpenRouter web search; verify citations/sources behavior.

## 12. Acceptance Tests

- Saying “Leonardo” triggers listening within 200 ms.
- 10 consecutive interactions without crash.
- Barge-in stops speaking within 150 ms of speech detection.
- Web query (e.g., “What happened in tech news today?”) returns an up-to-date response grounded in web results.
- Follow-up question within configured window is treated as continuation.

## 13. Risks

- Echo/feedback on a single machine (mic hears speaker). Mitigation: headphones, or enable system echo cancellation / use AEC-friendly audio devices.
- Wake word false positives. Mitigation: threshold tuning, push-to-talk fallback toggle.
- Latency variability from OpenRouter/provider routing. Mitigation: warm-up, provider selection, streaming.

## 14. Deliverables

- Repo with:
  - `assistant/` core state machine
  - `audio/` capture + VAD + playback + barge-in
  - `stt/` whisper integration
  - `llm/` OpenRouter client (warm-up + web-search)
  - `tts/` say/piper
  - `config.yaml`
  - `metrics.jsonl`
- Short README with setup instructions and usage.

