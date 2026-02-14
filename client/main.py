"""Leonardo Voice Assistant -- Client entry point.

Runs on the Raspberry Pi. Handles wake word detection, audio capture,
VAD, earcons, and TTS playback. Connects to the server via WebSocket
for STT/LLM/TTS processing.
"""

import logging
import signal
import sys
from pathlib import Path

import yaml

from client.audio.capture import AudioCapture
from client.audio.playback import AudioPlayer
from client.audio.vad import VoiceActivityDetector, UtteranceDetector
from client.wake.detector import WakeWordDetector
from client.connection import ServerConnection
from client.chunk_player import ChunkPlayer
from client.state_machine import ClientStateMachine


def load_config(path: str | None = None) -> dict:
    """Load client config from YAML file."""
    if path is None:
        path = str(Path(__file__).parent / "config.yaml")

    config_path = Path(path)
    if not config_path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Leonardo Voice Assistant Client")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--server", type=str, default=None, help="Server URL (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Build server URL
    server_cfg = config.get("server", {})
    if args.server:
        server_url = args.server
    else:
        host = server_cfg.get("host", "localhost")
        port = server_cfg.get("port", 8765)
        server_url = f"ws://{host}:{port}/ws"

    # Initialize components
    capture = AudioCapture(config["audio"])
    player = AudioPlayer(config["audio"]["sample_rate"])
    vad = VoiceActivityDetector(config["vad"], config["audio"]["sample_rate"])
    utterance_detector = UtteranceDetector(config["vad"])
    wake_detector = WakeWordDetector(config["wake"])
    connection = ServerConnection(
        server_url,
        reconnect_min_s=server_cfg.get("reconnect_min_s", 1.0),
        reconnect_max_s=server_cfg.get("reconnect_max_s", 30.0),
        offline_send_buffer_size=server_cfg.get("offline_send_buffer_size", 200),
        offline_send_ttl_s=server_cfg.get("offline_send_ttl_s", 5.0),
    )
    chunk_player = ChunkPlayer(player)

    machine = ClientStateMachine(
        config=config,
        capture=capture,
        player=player,
        vad=vad,
        utterance_detector=utterance_detector,
        wake_detector=wake_detector,
        connection=connection,
        chunk_player=chunk_player,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def shutdown(signum, frame):
        print("\n\033[32mShutting down...\033[0m")
        machine.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Connect to server
    print(f"\033[32mConnecting to server at {server_url}...\033[0m")
    connection.start()
    if not connection.wait_connected(timeout=30):
        print("\033[31mFailed to connect to server. Starting anyway (will reconnect).\033[0m")

    print("\033[32mLeonardo Voice Assistant (client) starting...\033[0m")
    try:
        machine.run()
    finally:
        capture.stop()
        connection.stop()
        print("\033[32mGoodbye.\033[0m")


if __name__ == "__main__":
    main()
