"""Short notification sounds (earcons) for state transitions."""

import numpy as np

from audio.playback import AudioPlayer


def generate_earcon(
    frequency: float = 880,
    duration_s: float = 0.15,
    volume: float = 0.3,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a sine-wave chime with a smooth envelope."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Apply a smooth fade-in / fade-out envelope
    envelope = np.ones_like(t)
    fade_len = int(sample_rate * 0.02)  # 20ms fade
    if fade_len > 0 and fade_len * 2 < len(t):
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    return (tone * envelope * volume).astype(np.float32)


def play_earcon(player: AudioPlayer, earcon_config: dict, sample_rate: int = 16000) -> None:
    """Generate and play a notification chime."""
    audio = generate_earcon(
        frequency=earcon_config["frequency"],
        duration_s=earcon_config["duration_s"],
        volume=earcon_config["volume"],
        sample_rate=sample_rate,
    )
    player.play(audio, sample_rate=sample_rate)
