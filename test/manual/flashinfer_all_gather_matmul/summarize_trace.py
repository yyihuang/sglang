#!/usr/bin/env python3
"""Find an exact backend kernel symbol in SGLang GPU activity traces."""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--kernel-regex", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(args.kernel_regex)

    trace_paths = sorted(
        path
        for path in args.trace_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    )
    matches = Counter()
    total_duration_us = 0.0
    gpu_kernel_events = 0
    for path in trace_paths:
        payload = read_json(path)
        events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
        for event in events:
            category = str(event.get("cat", "")).lower()
            if "kernel" not in category:
                continue
            gpu_kernel_events += 1
            name = str(event.get("name", ""))
            if pattern.search(name):
                matches[name] += 1
                total_duration_us += float(event.get("dur", 0.0))
    if not matches:
        raise RuntimeError(
            f"No GPU kernel event matched {args.kernel_regex!r} in {len(trace_paths)} traces"
        )
    result = {
        "kernel_regex": args.kernel_regex,
        "trace_files": [str(path) for path in trace_paths],
        "gpu_kernel_events": gpu_kernel_events,
        "matched_launches": sum(matches.values()),
        "matched_duration_us": total_duration_us,
        "matched_symbols": dict(matches.most_common()),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
