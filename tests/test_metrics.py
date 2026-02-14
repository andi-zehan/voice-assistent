from pathlib import Path

from server.assistant.metrics import MetricsLogger


def test_flush_interval_is_coerced_to_one(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "metrics.jsonl"
    logger = MetricsLogger({"enabled": True, "file": str(log_path), "flush_interval": 0})

    logger.log("event_a", value=1)
    logger.flush()

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_write_failure_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    logger = MetricsLogger({"enabled": True, "file": str(tmp_path / "metrics.jsonl"), "flush_interval": 1})

    def _broken_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _broken_open)

    logger.log("event_a", value=1)
    logger.flush()


def test_serialization_failure_drops_event_without_crashing(tmp_path: Path) -> None:
    logger = MetricsLogger({"enabled": True, "file": str(tmp_path / "metrics.jsonl"), "flush_interval": 1})

    logger.log("event_a", value=object())
    logger.flush()

    path = tmp_path / "metrics.jsonl"
    if path.exists():
        assert path.read_text().strip() == ""
