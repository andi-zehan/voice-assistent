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

During a response, simply start speaking. Playback stops and the assistant listens to your new utterance. **Use headphones** to avoid echo-triggered false barge-ins.
