"""Reduce CUDA profiler kernel intervals without counting launch gaps."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def _union_us(spans: list[tuple[float, float]]) -> float:
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    trace_files = sorted(args.trace_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise RuntimeError(f"no profiler traces found in {args.trace_dir}")

    intervals: dict[int, list[tuple[float, float]]] = {}
    kernel_sum_us: dict[int, float] = {}
    event_counts: dict[int, int] = {}
    for path in trace_files:
        rank_match = re.search(r"-TP-(\d+)(?:-|\.)", path.name)
        fallback_device = int(rank_match.group(1)) if rank_match else None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        events = payload.get(
            "traceEvents", payload if isinstance(payload, list) else []
        )
        for event in events:
            if event.get("ph") != "X" or str(event.get("cat", "")).lower() != "kernel":
                continue
            duration = float(event.get("dur", 0.0))
            if duration <= 0.0:
                continue
            device_value = (event.get("args") or {}).get("device", fallback_device)
            if device_value is None:
                raise RuntimeError(f"kernel event has no device or TP rank: {path}")
            device = int(device_value)
            start = float(event["ts"])
            intervals.setdefault(device, []).append((start, start + duration))
            kernel_sum_us[device] = kernel_sum_us.get(device, 0.0) + duration
            event_counts[device] = event_counts.get(device, 0) + 1

    if not intervals:
        raise RuntimeError("no CUDA kernel intervals found")

    all_intervals = [span for spans in intervals.values() for span in spans]
    devices = {
        str(device): {
            "kernel_count": event_counts[device],
            "active_union_ms": _union_us(spans) / 1000.0,
            "kernel_sum_ms": kernel_sum_us[device] / 1000.0,
        }
        for device, spans in sorted(intervals.items())
    }
    result = {
        "metric": "cupti_gpu_kernel_active_union_ms",
        "source": "torch.profiler.ProfilerActivity.CUDA",
        "inter_kernel_launch_gaps_excluded": True,
        "devices": devices,
        "aggregate_device_active_union_ms": sum(
            row["active_union_ms"] for row in devices.values()
        ),
        "cross_device_active_union_ms": _union_us(all_intervals) / 1000.0,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SGLANG_DSV4_ACTIVE_UNION=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
