#!/usr/bin/env python3
"""Verify that the pinned GSM8K evaluation completed the exact sample count."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    score = metrics.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError(f"Invalid GSM8K score: {score}")
    evaluated = args.report.read_text().count("<hr>")
    if evaluated != args.expected_examples:
        raise RuntimeError(
            f"GSM8K report contains {evaluated} examples, expected {args.expected_examples}"
        )
    result = {
        "evaluated_examples": evaluated,
        "expected_examples": args.expected_examples,
        "score": score,
        "metrics_sha256": sha256_file(args.metrics),
        "report_sha256": sha256_file(args.report),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
