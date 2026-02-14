"""Conversation session with history management."""


class Session:
    """Maintains conversation history with automatic trimming."""

    def __init__(self, conversation_config: dict):
        self._max_turns = conversation_config["max_turns"]
        self._max_tokens_budget = conversation_config["max_tokens_budget"]
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def add_user_message(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return the conversation history (without system prompt)."""
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def _trim(self) -> None:
        """Trim history to stay within turn and token budget limits."""
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        while len(self._history) > 2:
            total_chars = sum(len(m["content"]) for m in self._history)
            estimated_tokens = total_chars / 4
            if estimated_tokens <= self._max_tokens_budget:
                break
            self._history = self._history[2:]
