#!/usr/bin/env python3
"""Run a long-duration soak session and summarize robustness metrics.

This script can optionally launch the assistant process and monitor events written
into ``metrics.jsonl``. It provides a pass/fail exit code based on configurable
thresholds so it can be used in manual validation and CI-style smoke checks.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SoakStats:
    events_total: int = 0
    pipeline_errors: int = 0
    listening_timeouts: int = 0
    barge_in_events: int = 0
    audio_frame_drops: int = 0
    wake_events: int = 0
    interactions: int = 0
    interaction_latencies: list[float] = field(default_factory=list)

    def add_event(self, event: dict) -> None:
        self.events_total += 1
        kind = event.get("event")

        if kind == "pipeline_error":
            self.pipeline_errors += 1
        elif kind == "listening_timeout":
            self.listening_timeouts += 1
        elif kind == "barge_in":
            self.barge_in_events += 1
        elif kind == "wake_detected":
            self.wake_events += 1
        elif kind == "audio_frame_drop":
            self.audio_frame_drops += int(event.get("dropped_frames", 0) or 0)
        elif kind == "interaction_complete":
            self.interactions += 1
            latency = event.get("total_elapsed_s")
            if isinstance(latency, (int, float)):
                self.interaction_latencies.append(float(latency))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness soak checks from metrics JSONL")
    parser.add_argument("--metrics-file", default="metrics.jsonl", help="Path to metrics JSONL file")
    parser.add_argument("--duration-s", type=int, default=900, help="Monitoring duration in seconds")
    parser.add_argument("--poll-s", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--status-every-s", type=int, default=30, help="Status print cadence in seconds")
    parser.add_argument(
        "--command",
        default="",
        help="Optional command to launch while monitoring (example: 'python3 main.py')",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include existing metrics lines; otherwise monitor only new lines",
    )

    # Pass/fail thresholds
    parser.add_argument("--min-interactions", type=int, default=3)
    parser.add_argument("--max-pipeline-errors", type=int, default=0)
    parser.add_argument("--max-listening-timeouts", type=int, default=50)
    parser.add_argument("--max-audio-frame-drops", type=int, default=2000)
    parser.add_argument("--max-p95-latency-s", type=float, default=10.0)

    return parser.parse_args()


def read_new_events(path: Path, offset: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], offset

    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        new_offset = f.tell()

    return events, new_offset


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = rank - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


def print_status(stats: SoakStats, elapsed_s: float) -> None:
    p95 = percentile(stats.interaction_latencies, 0.95)
    print(
        "[soak]"
        f" t={elapsed_s:6.1f}s"
        f" events={stats.events_total}"
        f" interactions={stats.interactions}"
        f" pipeline_errors={stats.pipeline_errors}"
        f" listening_timeouts={stats.listening_timeouts}"
        f" drops={stats.audio_frame_drops}"
        f" p95={p95:.2f}s"
    )


def build_summary(stats: SoakStats) -> str:
    p50 = percentile(stats.interaction_latencies, 0.50)
    p95 = percentile(stats.interaction_latencies, 0.95)
    p99 = percentile(stats.interaction_latencies, 0.99)
    return (
        "\nSoak Summary\n"
        f"- events_total: {stats.events_total}\n"
        f"- wake_events: {stats.wake_events}\n"
        f"- interactions: {stats.interactions}\n"
        f"- pipeline_errors: {stats.pipeline_errors}\n"
        f"- listening_timeouts: {stats.listening_timeouts}\n"
        f"- barge_in_events: {stats.barge_in_events}\n"
        f"- audio_frame_drops: {stats.audio_frame_drops}\n"
        f"- latency_p50_s: {p50:.3f}\n"
        f"- latency_p95_s: {p95:.3f}\n"
        f"- latency_p99_s: {p99:.3f}\n"
    )


def evaluate_thresholds(stats: SoakStats, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if stats.interactions < args.min_interactions:
        failures.append(
            f"interactions {stats.interactions} < min_interactions {args.min_interactions}"
        )
    if stats.pipeline_errors > args.max_pipeline_errors:
        failures.append(
            f"pipeline_errors {stats.pipeline_errors} > max_pipeline_errors {args.max_pipeline_errors}"
        )
    if stats.listening_timeouts > args.max_listening_timeouts:
        failures.append(
            f"listening_timeouts {stats.listening_timeouts} > max_listening_timeouts {args.max_listening_timeouts}"
        )
    if stats.audio_frame_drops > args.max_audio_frame_drops:
        failures.append(
            f"audio_frame_drops {stats.audio_frame_drops} > max_audio_frame_drops {args.max_audio_frame_drops}"
        )
    p95 = percentile(stats.interaction_latencies, 0.95)
    if p95 > args.max_p95_latency_s:
        failures.append(f"latency_p95_s {p95:.3f} > max_p95_latency_s {args.max_p95_latency_s}")
    return failures


def start_process(command: str) -> subprocess.Popen | None:
    if not command:
        return None
    argv = shlex.split(command)
    if not argv:
        return None
    print(f"[soak] launching: {command}")
    return subprocess.Popen(argv)


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_file)

    offset = 0
    if metrics_path.exists() and not args.include_existing:
        offset = metrics_path.stat().st_size

    proc = start_process(args.command)
    stats = SoakStats()

    start = time.monotonic()
    last_status = start

    print(
        f"[soak] monitoring '{metrics_path}' for {args.duration_s}s "
        f"(poll={args.poll_s}s, include_existing={args.include_existing})"
    )

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration_s:
                break

            events, offset = read_new_events(metrics_path, offset)
            for event in events:
                stats.add_event(event)

            if now - last_status >= args.status_every_s:
                print_status(stats, elapsed)
                last_status = now

            if proc is not None and proc.poll() is not None:
                print(f"[soak] monitored command exited early with code {proc.returncode}")
                break

            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("[soak] interrupted by user")
    finally:
        stop_process(proc)

    # Final read in case the last cycle wrote metrics.
    events, _ = read_new_events(metrics_path, offset)
    for event in events:
        stats.add_event(event)

    print(build_summary(stats))
    failures = evaluate_thresholds(stats, args)
    if failures:
        print("Soak Result: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Soak Result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
