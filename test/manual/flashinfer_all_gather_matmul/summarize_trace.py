#!/usr/bin/env python3
"""Find an exact backend kernel symbol in SGLang GPU activity traces."""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--expected-kernel-symbol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_paths = sorted(
        path
        for path in args.trace_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    )
    if len(trace_paths) != 4:
        raise RuntimeError(f"Expected exactly four TP traces, found {len(trace_paths)}")
    rank_paths = {}
    for path in trace_paths:
        rank_match = re.search(r"(?:^|-)TP-([0-3])(?:-|\.)", path.name)
        if not rank_match:
            raise RuntimeError(f"Trace filename has no TP rank: {path}")
        rank = int(rank_match.group(1))
        if rank in rank_paths:
            raise RuntimeError(f"Multiple traces found for TP rank {rank}")
        rank_paths[rank] = path
    if set(rank_paths) != {0, 1, 2, 3}:
        raise RuntimeError(f"Trace ranks are not exactly 0..3: {sorted(rank_paths)}")

    rank_results = {}
    all_matches = Counter()
    for rank, path in sorted(rank_paths.items()):
        payload = read_json(path)
        events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
        matches = Counter()
        total_duration_us = 0.0
        gpu_kernel_events = 0
        for event in events:
            category = str(event.get("cat", "")).lower()
            if "kernel" not in category:
                continue
            gpu_kernel_events += 1
            name = str(event.get("name", ""))
            if name == args.expected_kernel_symbol:
                matches[name] += 1
                total_duration_us += float(event.get("dur", 0.0))
        if not matches:
            raise RuntimeError(
                f"TP rank {rank} has no exact GPU kernel {args.expected_kernel_symbol!r}"
            )
        all_matches.update(matches)
        rank_results[str(rank)] = {
            "trace_path": str(path),
            "trace_sha256": sha256_file(path),
            "gpu_kernel_events": gpu_kernel_events,
            "matched_launches": sum(matches.values()),
            "matched_duration_us_route_evidence_only": total_duration_us,
            "matched_symbols": dict(matches.most_common()),
        }
    result = {
        "expected_kernel_symbol": args.expected_kernel_symbol,
        "rank_count": 4,
        "ranks": rank_results,
        "matched_launches": sum(all_matches.values()),
        "matched_symbols": dict(all_matches.most_common()),
        "timing_use": "route evidence only; not a performance metric",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
