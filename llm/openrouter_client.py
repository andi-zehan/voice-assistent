"""OpenRouter LLM client with streaming SSE support."""

import json
import os
import time
import threading

import requests


class OpenRouterClient:
    """Streaming HTTP client for OpenRouter's chat completions API."""

    def __init__(self, llm_config: dict):
        self._model = llm_config["model"]
        self._api_base = llm_config["api_base"]
        self._max_tokens = llm_config["max_tokens"]
        self._temperature = llm_config["temperature"]
        self._web_search = llm_config.get("web_search", False)
        self._warmup_enabled = llm_config.get("warmup_enabled", True)
        self._timeout = llm_config.get("timeout_s", 30)

        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            print("WARNING: OPENROUTER_API_KEY not set")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/leonardo-assistant",
            "X-Title": "Leonardo Voice Assistant",
        }

    def warmup(self) -> None:
        """Fire-and-forget minimal request to warm up the API connection.

        Sends a tiny streaming request (max_tokens=1) in a background thread.
        The response is discarded — this just reduces TTFT for the real request.
        """
        if not self._warmup_enabled:
            return

        def _do_warmup():
            try:
                payload = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": True,
                }
                resp = requests.post(
                    f"{self._api_base}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    stream=True,
                    timeout=10,
                )
                resp.close()
            except Exception:
                pass  # Warmup failure is not critical

        threading.Thread(target=_do_warmup, daemon=True).start()

    def chat(self, messages: list[dict]) -> dict:
        """Send a chat completion request with streaming SSE.

        Returns dict with keys: text, model, elapsed_s, ttft_s
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
        }

        if self._web_search:
            payload["plugins"] = [{"id": "web"}]

        t0 = time.monotonic()
        ttft = None
        full_text = ""
        model_used = self._model

        resp = requests.post(
            f"{self._api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self._timeout,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # Strip "data: " prefix
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "model" in data:
                model_used = data["model"]

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if ttft is None:
                    ttft = time.monotonic() - t0
                full_text += content

        resp.close()
        elapsed = time.monotonic() - t0

        return {
            "text": full_text.strip(),
            "model": model_used,
            "elapsed_s": elapsed,
            "ttft_s": ttft or elapsed,
        }
