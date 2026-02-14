"""JSONL event logger for interaction metrics."""

import json
import threading
import time
from pathlib import Path


class MetricsLogger:
    """Thread-safe JSONL logger with periodic flushing."""

    def __init__(self, metrics_config: dict):
        self._enabled = metrics_config.get("enabled", True)
        self._file_path = Path(metrics_config.get("file", "metrics.jsonl"))
        try:
            flush_interval = int(metrics_config.get("flush_interval", 10))
        except (TypeError, ValueError):
            flush_interval = 10
        self._flush_interval = max(1, flush_interval)

        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._event_count = 0
        self._write_error_count = 0
        self._last_warn_s = 0.0
        self._warn_interval_s = 30.0

        if self._enabled:
            try:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._enabled = False
                self._warn_write_error("metrics path is not writable; disabling metrics")

    def log(self, event_type: str, **data) -> None:
        """Log an event with a timestamp."""
        if not self._enabled:
            return

        entry = {
            "timestamp": time.time(),
            "event": event_type,
            **data,
        }
        try:
            line = json.dumps(entry, default=lambda o: float(o))
        except (TypeError, ValueError, OverflowError):
            self._warn_write_error("metrics serialization failed; dropping event")
            return

        with self._lock:
            self._buffer.append(line)
            self._event_count += 1
            if self._event_count % self._flush_interval == 0:
                self._flush_locked_safe()

    def flush(self) -> None:
        """Write all buffered events to disk."""
        with self._lock:
            self._flush_locked_safe()

    def _flush_locked(self) -> None:
        """Internal flush — must be called with lock held."""
        if not self._buffer:
            return
        with open(self._file_path, "a") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()

    def _flush_locked_safe(self) -> None:
        """Flush and absorb filesystem errors so metrics never crash runtime."""
        try:
            self._flush_locked()
        except (OSError, ValueError):
            # Drop buffered events to avoid unbounded memory growth.
            self._buffer.clear()
            self._write_error_count += 1
            self._warn_write_error("metrics flush failed; dropping buffered events")

    def _warn_write_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn_s < self._warn_interval_s:
            return
        self._last_warn_s = now
        print(f"\033[31mWARNING: {message}\033[0m")
