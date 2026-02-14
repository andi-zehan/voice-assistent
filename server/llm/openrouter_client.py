"""OpenRouter LLM client with streaming SSE support."""

import json
import os
import time
import threading
import random

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
        try:
            max_retries = int(llm_config.get("max_retries", 2))
        except (TypeError, ValueError):
            max_retries = 2
        self._max_retries = max(0, max_retries)
        try:
            retry_base_delay_s = float(llm_config.get("retry_base_delay_s", 0.25))
        except (TypeError, ValueError):
            retry_base_delay_s = 0.25
        self._retry_base_delay_s = max(0.05, retry_base_delay_s)

        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            print("\033[31mWARNING: OPENROUTER_API_KEY not set\033[0m")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/leonardo-assistant",
            "X-Title": "Jarvis Voice Assistant",
        }

    def warmup(self) -> None:
        """Fire-and-forget minimal request to warm up the API connection."""
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
                pass

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
        attempts = self._max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            resp = None
            ttft = None
            full_text = ""
            model_used = self._model

            try:
                resp = requests.post(
                    f"{self._api_base}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    stream=True,
                    timeout=self._timeout,
                )
                if resp.status_code >= 400:
                    should_retry = self._should_retry_status(resp.status_code)
                    if should_retry and attempt < attempts - 1:
                        self._sleep_before_retry(attempt)
                        continue
                    resp.raise_for_status()

                resp.encoding = "utf-8"
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
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

                elapsed = time.monotonic() - t0
                return {
                    "text": full_text.strip(),
                    "model": model_used,
                    "elapsed_s": elapsed,
                    "ttft_s": ttft or elapsed,
                }

            except requests.RequestException as exc:
                last_error = exc
                retryable = True
                if isinstance(exc, requests.HTTPError):
                    status = exc.response.status_code if exc.response is not None else None
                    retryable = bool(status and self._should_retry_status(status))
                if (not retryable) or (attempt >= attempts - 1):
                    raise
                self._sleep_before_retry(attempt)
            finally:
                if resp is not None:
                    resp.close()

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter chat failed without exception")

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    def _sleep_before_retry(self, attempt: int) -> None:
        base = self._retry_base_delay_s * (2 ** attempt)
        jitter = random.uniform(0.0, base * 0.25)
        time.sleep(base + jitter)
