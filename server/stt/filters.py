"""STT hallucination detection filters.

Extracted from the monolithic state_machine.py for server-side use.
"""

# Common Whisper hallucinations on silence/noise
_HALLUCINATION_PHRASES = {
    # English
    "thank you for watching",
    "thanks for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "see you next time",
    "bye bye",
    "thank you",
    "thanks for listening",
    "the end",
    "you",
    "i'm sorry",
    # German
    "danke fürs zuschauen",
    "danke für's zuschauen",
    "vielen dank fürs zuschauen",
    "bis zum nächsten mal",
    "tschüss",
    "untertitel von stephanie geiges",
    "untertitel der amara.org-community",
    "untertitel im auftrag des zdf für funk",
}


def check_hallucination(
    text: str,
    no_speech_prob: float,
    avg_logprob: float,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
) -> tuple[bool, str]:
    """Return (rejected, reason) if the transcript looks like a hallucination."""
    if no_speech_prob >= no_speech_threshold:
        return True, f"no_speech_prob={no_speech_prob:.2f}"

    if avg_logprob < logprob_threshold:
        return True, f"avg_logprob={avg_logprob:.2f}"

    if text.strip().lower().rstrip(".!?,") in _HALLUCINATION_PHRASES:
        return True, "hallucination_blocklist"

    return False, ""
