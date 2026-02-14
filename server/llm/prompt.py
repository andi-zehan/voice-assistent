"""System prompt and message formatting for the LLM."""

import re

_BASE_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise voice assistant. "
    "Your responses will be spoken aloud by a text-to-speech engine. "
    "Be concise and to the point. "
    "NEVER include citations, reference numbers, URLs, links, footnotes, "
    "source attributions, or any markup in your responses. "
    "Do not use markdown, bullet points, numbered lists, or code blocks. "
    "Just answer naturally as a human would in a spoken conversation. "
    "If you don't know something, say so honestly. "
    "Even when web search is used, never mention sources or citations."
)

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
}

DEFAULT_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT


def get_system_prompt(language: str | None = None) -> str:
    """Return the system prompt, optionally tailored to *language*."""
    if not language or language == "en":
        return _BASE_SYSTEM_PROMPT

    lang_name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"{_BASE_SYSTEM_PROMPT} "
        f"The user is speaking in {lang_name}. "
        f"Always respond in {lang_name} unless the user explicitly asks "
        "for a different language (for example, when requesting a translation)."
    )


def clean_for_tts(text: str) -> str:
    """Strip citations, URLs, markdown, and other non-speakable artifacts."""
    text = re.sub(r'\uE200.*?\uE201', '', text, flags=re.DOTALL)
    text = re.sub(r'[\u3010\u3016][^\u3011\u3017]+[\u3011\u3017]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[\d+(?:[,\s]*\d+)*\]', '', text)
    text = re.sub(r'\[(?:source|citation|ref)\w*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:source|sources|citation|citations|ref\w*|quelle|quellen)[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\^(?:\d+|source|ref\w*)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((?:source|sources|citation|citations|reference|references|quelle|quellen)\s*:[^)]+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?im)^\s*(?:sources?|references?|citations?|quellen?)\s*:\s*$', '', text)
    text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        if re.match(r'(?i)^(?:sources?|references?|citations?|quellen?)\s*:?\s*$', stripped):
            continue
        if re.match(r'^(?:\[\d+\]|\d+[.)])\s*$', stripped):
            continue
        if re.match(r'(?i)^(?:\[\d+\]|\d+[.)])\s*(?:https?://\S+|www\.\S+)\s*$', stripped):
            continue
        if re.match(r'(?i)^(?:https?://\S+|www\.\S+)\s*$', stripped):
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?]){2,}', r'\1', text)
    return text.strip()


def build_messages(
    system_prompt: str,
    history: list[dict],
    user_text: str,
) -> list[dict]:
    """Build the messages list for the LLM API call."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
