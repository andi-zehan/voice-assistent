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
        self._flush_interval = metrics_config.get("flush_interval", 10)

        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._event_count = 0

    def log(self, event_type: str, **data) -> None:
        """Log an event with a timestamp."""
        if not self._enabled:
            return

        entry = {
            "timestamp": time.time(),
            "event": event_type,
            **data,
        }
        line = json.dumps(entry, default=lambda o: float(o))

        with self._lock:
            self._buffer.append(line)
            self._event_count += 1
            if self._event_count % self._flush_interval == 0:
                self._flush_locked()

    def flush(self) -> None:
        """Write all buffered events to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush — must be called with lock held."""
        if not self._buffer:
            return
        with open(self._file_path, "a") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()
