"""System prompt and message formatting for the LLM."""

import re

DEFAULT_SYSTEM_PROMPT = (
    "You are Leonardo, a helpful and concise voice assistant. "
    "Your responses will be spoken aloud by a text-to-speech engine. "
    "Be concise and to the point. "
    "NEVER include citations, reference numbers, URLs, links, footnotes, "
    "source attributions, or any markup in your responses. "
    "Do not use markdown, bullet points, numbered lists, or code blocks. "
    "Just answer naturally as a human would in a spoken conversation. "
    "If you don't know something, say so honestly."
)


def clean_for_tts(text: str) -> str:
    """Strip citations, URLs, markdown, and other non-speakable artifacts."""
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove citation markers like [1], [2, 3], [source], etc.
    text = re.sub(r'\[\d+(?:[,\s]*\d+)*\]', '', text)
    text = re.sub(r'\[(?:source|citation|ref)\w*\]', '', text, flags=re.IGNORECASE)
    # Remove footnote-style markers like ¹ ² ³
    text = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', text)
    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bullet point markers
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def build_messages(
    system_prompt: str,
    history: list[dict],
    user_text: str,
) -> list[dict]:
    """Build the messages list for the LLM API call.

    Args:
        system_prompt: The system-level instruction.
        history: Previous conversation turns [{"role": ..., "content": ...}, ...].
        user_text: The latest user utterance.

    Returns:
        Full messages list including system, history, and the new user message.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
