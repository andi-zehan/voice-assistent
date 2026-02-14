"""Short notification sounds (earcons) for state transitions."""

import numpy as np

from client.audio.playback import AudioPlayer


def generate_tone(
    frequency: float = 880,
    duration_s: float = 0.15,
    volume: float = 0.3,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a sine-wave tone with a smooth fade-in/fade-out envelope."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    envelope = np.ones_like(t)
    fade_len = int(sample_rate * 0.02)
    if fade_len > 0 and fade_len * 2 < len(t):
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    return (tone * envelope * volume).astype(np.float32)


def generate_earcon(name: str, sample_rate: int = 16000, volume: float = 0.3) -> np.ndarray:
    """Generate a named earcon.

    Supported names:
        wake       -- rising chime on wake word detection (A5, 150ms)
        heard      -- short low confirmation when utterance captured (A4, 100ms)
        ready      -- two rising pips when follow-up window opens (E5->A5)
        goodbye    -- descending tone when session ends (A5->A4, 200ms)
        error      -- double low buzz on pipeline error (A3, 80ms x2)
    """
    if name == "wake":
        return generate_tone(880, 0.15, volume, sample_rate)

    if name == "heard":
        return generate_tone(440, 0.10, volume, sample_rate)

    if name == "ready":
        pip1 = generate_tone(660, 0.08, volume, sample_rate)
        gap = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
        pip2 = generate_tone(880, 0.08, volume, sample_rate)
        return np.concatenate([pip1, gap, pip2])

    if name == "goodbye":
        t = np.linspace(0, 0.20, int(sample_rate * 0.20), endpoint=False)
        freq = np.linspace(880, 440, len(t))
        tone = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate)
        envelope = np.ones_like(t)
        fade_len = int(sample_rate * 0.02)
        if fade_len > 0 and fade_len * 2 < len(t):
            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        return (tone * envelope * volume).astype(np.float32)

    if name == "error":
        buzz1 = generate_tone(220, 0.08, volume, sample_rate)
        gap = np.zeros(int(sample_rate * 0.06), dtype=np.float32)
        buzz2 = generate_tone(220, 0.08, volume, sample_rate)
        return np.concatenate([buzz1, gap, buzz2])

    raise ValueError(f"Unknown earcon: {name!r}")


def play_earcon(player: AudioPlayer, earcon_config: dict, sample_rate: int = 16000) -> None:
    """Generate and play the wake earcon (legacy helper)."""
    audio = generate_tone(
        frequency=earcon_config["frequency"],
        duration_s=earcon_config["duration_s"],
        volume=earcon_config["volume"],
        sample_rate=sample_rate,
    )
    player.play(audio, sample_rate=sample_rate)


def play_named_earcon(
    player: AudioPlayer, name: str, sample_rate: int = 16000, volume: float = 0.3,
) -> None:
    """Generate and play a named earcon."""
    audio = generate_earcon(name, sample_rate=sample_rate, volume=volume)
    player.play(audio, sample_rate=sample_rate)
