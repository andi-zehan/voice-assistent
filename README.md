# Leonardo Voice Assistant

A macOS desktop voice assistant prototype triggered by the wake word "Leonardo" (currently using "hey jarvis" pre-trained model).

## Pipeline

Wake word detection → Speech capture → VAD/end-of-utterance → STT (Whisper) → LLM (OpenRouter) → TTS (macOS `say`) → Playback

Supports barge-in (interrupt during response) and follow-up conversation without repeating the wake word.

## Prerequisites

- macOS (uses `say` for TTS)
- Python 3.10+
- PortAudio and libsndfile:

```bash
brew install portaudio libsndfile
```

## Setup

```bash
# Clone and install
cd leonardo_v1
pip install -e .

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
python main.py
```

1. Wait for "PASSIVE" state in console
2. Say the wake word → hear a chime
3. Ask your question → hear the response
4. Follow up within 4 seconds without the wake word, or wait for it to return to passive

## Configuration

All settings are in `config.yaml` — audio devices, VAD sensitivity, STT model size, LLM model, TTS voice, conversation history limits, etc.

## Barge-in

Barge-in is implemented but **disabled by default** (`vad.barge_in_enabled: false`) to avoid echo-triggered false interrupts without AEC.

To enable it:

1. Set `vad.barge_in_enabled: true` in `config.yaml`
2. Keep `vad.barge_in_grace_s` and `vad.barge_in_frames` conservative
3. Use headphones or an AEC-capable setup

During a response, start speaking and playback will stop once speech onset is confirmed.

## Privacy And Metrics

Current development defaults keep full text logging on: `metrics.log_transcripts: true` and `metrics.log_llm_text: true`, so raw speech transcripts and assistant response text are written to `metrics.jsonl`.

For safer handling in shared environments, set these keys to `false`.
Those same flags also control whether transcript/response text is printed verbatim in the terminal.
