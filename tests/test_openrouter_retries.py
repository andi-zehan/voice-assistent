import os

import pytest
import requests

from llm.openrouter_client import OpenRouterClient


class FakeResponse:
    def __init__(self, status_code: int, lines: list[str] | None = None):
        self.status_code = status_code
        self._lines = lines or []
        self.encoding = None
        self.closed = False

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def close(self) -> None:
        self.closed = True


def _client() -> OpenRouterClient:
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    return OpenRouterClient(
        {
            "model": "openai/gpt-5-chat",
            "api_base": "https://openrouter.ai/api/v1",
            "max_tokens": 32,
            "temperature": 0.1,
            "timeout_s": 1,
            "max_retries": 2,
            "retry_base_delay_s": 0.01,
        }
    )


def test_retries_on_timeout_then_succeeds(monkeypatch) -> None:
    client = _client()
    calls = {"n": 0}

    success_lines = [
        'data: {"model":"openai/gpt-5-chat","choices":[{"delta":{"content":"hello"}}]}',
        "data: [DONE]",
    ]

    def _post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("temporary timeout")
        return FakeResponse(200, success_lines)

    monkeypatch.setattr("llm.openrouter_client.requests.post", _post)
    monkeypatch.setattr("llm.openrouter_client.time.sleep", lambda *_: None)

    out = client.chat([{"role": "user", "content": "hi"}])

    assert out["text"] == "hello"
    assert calls["n"] == 2


def test_does_not_retry_on_401(monkeypatch) -> None:
    client = _client()
    calls = {"n": 0}

    def _post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(401)

    monkeypatch.setattr("llm.openrouter_client.requests.post", _post)
    monkeypatch.setattr("llm.openrouter_client.time.sleep", lambda *_: None)

    with pytest.raises(requests.HTTPError):
        client.chat([{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
