"""Jarvis Voice Assistant — entry point."""

import signal
import sys
from pathlib import Path

import yaml

from assistant.state_machine import StateMachine
from audio.capture import AudioCapture
from audio.playback import AudioPlayer
from audio.vad import VoiceActivityDetector, UtteranceDetector
from wake.detector import WakeWordDetector
from stt.whisper_stt import WhisperSTT
from llm.openrouter_client import OpenRouterClient
from tts import create_tts
from assistant.session import Session
from assistant.metrics import MetricsLogger


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Initialize components
    capture = AudioCapture(config["audio"])
    player = AudioPlayer(config["audio"]["sample_rate"])
    vad = VoiceActivityDetector(config["vad"], config["audio"]["sample_rate"])
    utterance_detector = UtteranceDetector(config["vad"])
    wake_detector = WakeWordDetector(config["wake"])
    stt = WhisperSTT(config["stt"])
    llm_client = OpenRouterClient(config["llm"])
    tts = create_tts(config["tts"])
    session = Session(config["conversation"])
    metrics = MetricsLogger(config["metrics"])

    machine = StateMachine(
        config=config,
        capture=capture,
        player=player,
        vad=vad,
        utterance_detector=utterance_detector,
        wake_detector=wake_detector,
        stt=stt,
        llm_client=llm_client,
        tts=tts,
        session=session,
        metrics=metrics,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def shutdown(signum, frame):
        print("\nShutting down...")
        machine.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Jarvis Voice Assistant starting...")
    try:
        machine.run()
    finally:
        capture.stop()
        metrics.flush()
        print("Goodbye.")


if __name__ == "__main__":
    main()
