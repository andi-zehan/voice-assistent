# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leonardo is a macOS desktop voice assistant prototype triggered by the wake word "Leonardo". It runs an end-to-end pipeline: wake word detection → speech capture → VAD/end-of-utterance → STT (Whisper) → LLM (OpenRouter with web search) → TTS → playback, with barge-in and follow-up conversation support.

## Tech Stack

- **Language**: Python
- **Audio I/O**: `sounddevice` or `pyaudio` — 16 kHz mono internal format
- **VAD**: WebRTC VAD
- **Wake word**: `openWakeWord`
- **STT**: `whisper.cpp` (CLI bindings) or `faster-whisper`
- **LLM**: OpenRouter HTTP API (streaming), with web search via `:online` model variant or web-search plugin
- **TTS**: macOS `say` (v1), Piper (v2)
- **Config**: `config.yaml`
- **Metrics**: `metrics.jsonl`

## Architecture

### State Machine

The assistant operates as a state machine with five states:
- **PASSIVE** — wake word detector running, waiting for "Leonardo"
- **LISTENING** — capturing speech after wake, VAD detecting end-of-utterance
- **THINKING** — running STT then LLM request
- **SPEAKING** — playing TTS audio
- **FOLLOW_UP** — short window (3-6s) for continuation without wake word; returns to PASSIVE on timeout

### Planned Directory Structure

```
assistant/    # Core state machine and orchestration
audio/        # Mic capture, VAD, playback, barge-in, ring buffer
stt/          # Whisper integration
llm/          # OpenRouter client (streaming, warm-up, web search)
tts/          # macOS say / Piper wrapper
config.yaml   # All configuration (devices, thresholds, API keys, model)
metrics.jsonl # Timestamped interaction logs
```

### Key Data Flow

1. Ring buffer continuously captures mic audio
2. Wake word triggers session start + LLM warm-up request (minimal 1-3 token streaming request, output discarded — reduces TTFT for real request)
3. VAD segments the utterance → PCM/WAV buffer
4. Whisper STT returns transcript
5. OpenRouter LLM request (with conversation context, web search enabled) → response text
6. TTS converts to audio → playback
7. During playback, VAD monitors mic for barge-in (stop playback, return to LISTENING)

### Barge-in

During SPEAKING state, VAD continues monitoring the mic. If user speech is detected, playback stops immediately and the system transitions to LISTENING.

### Conversation Context

Maintains a short conversation history (configurable max turns / token budget). Follow-up window allows continuation without repeating the wake word.

## Key Design Decisions (v1)

- Generic wake word detector is acceptable; defer custom "Leonardo" model training
- TTS uses macOS `say` — fastest integration path
- LLM responds fully before TTS starts (sentence-level streaming deferred to v2)
- Web search uses simplest supported OpenRouter online mode (`:online` preferred, plugin fallback)
- Echo cancellation handled by headphones or system AEC; no custom AEC in v1
