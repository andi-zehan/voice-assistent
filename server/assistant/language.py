"""Language helpers for response voice selection."""

_DE_CHARS = set("äöüßÄÖÜ")
_DE_STRONG = {
    "ich", "und", "der", "das", "ist", "ein", "eine", "nicht", "auf",
    "mit", "den", "dem", "sich", "von", "für", "aber", "wenn",
    "nur", "noch", "nach", "auch", "schon", "dann", "kann", "wir",
    "uns", "ihr", "wird", "oder", "sind", "bei", "haben", "hatte",
    "habe", "dir", "sehr", "hier", "diese", "dieser",
    "geht", "gibt", "bitte", "gerne", "danke", "jetzt", "kein",
    "keine", "mein", "meine", "dein", "immer", "dort", "denn", "weil",
}


def detect_response_language(text: str, fallback: str = "en") -> str:
    """Detect whether *text* is German or English for TTS voice selection."""
    if any(c in _DE_CHARS for c in text):
        return "de"

    words = {w.strip(".,!?;:\"'()[]") for w in text.lower().split()}
    if words & _DE_STRONG:
        return "de"

    normalized_fallback = (fallback or "en").lower()
    return normalized_fallback if normalized_fallback in {"en", "de"} else "en"
