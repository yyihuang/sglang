#!/usr/bin/env python3
"""Summarize three-arm accuracy, E2E metrics, route evidence, and speedups."""

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


def load(path):
    return json.loads(path.read_text())


def load_jsonl_last(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1])


def summarize_arm(root, name):
    path = root / name
    repetitions = [load_jsonl_last(path / f"serving-{i}.jsonl") for i in range(1, 4)]
    medians = {key: statistics.median(row[key] for row in repetitions) for key in METRICS}
    arm = {
        "environment": load(path / "environment.json"),
        "physical_seconds": int((path / "physical-seconds.txt").read_text()),
        "server_ready_seconds": int((path / "server-ready-seconds.txt").read_text()),
        "gsm8k": load(path / "gsm8k.json"),
        "serving_repetitions": repetitions,
        "serving_medians": medians,
    }
    if (path / "route-evidence.json").exists():
        arm["route_evidence"] = load(path / "route-evidence.json")
    if (path / "cake-kernel-evidence.json").exists():
        arm["cake_kernel_evidence"] = load(path / "cake-kernel-evidence.json")
    if (path / "parity-vs-native.json").exists():
        arm["parity_vs_native"] = load(path / "parity-vs-native.json")
    return arm


def ratios(reference, candidate):
    ref, got = reference["serving_medians"], candidate["serving_medians"]
    result = {}
    for key in ("request_throughput", "input_throughput", "output_throughput", "total_throughput"):
        result[key + "_speedup"] = got[key] / ref[key]
    for key in (
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
    ):
        result[key + "_speedup"] = ref[key] / got[key]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {name: summarize_arm(args.result_root, name) for name in ("native", "explicit", "candidate")}
    result = {
        "model_repo": "meta-llama/Llama-3.1-70B-Instruct",
        "model_revision": "1605565b47bb9346c5515c34102e054115b4f98b",
        "seed": 20260825,
        "input_contract": load(args.result_root / "input-contract.json"),
        "arms": arms,
        "fused_speedup_candidate_vs_explicit": ratios(arms["explicit"], arms["candidate"]),
        "net_candidate_vs_native": ratios(arms["native"], arms["candidate"]),
        "gsm8k_score_delta_candidate_minus_explicit": arms["candidate"]["gsm8k"]["score"] - arms["explicit"]["gsm8k"]["score"],
        "gsm8k_score_delta_candidate_minus_native": arms["candidate"]["gsm8k"]["score"] - arms["native"]["gsm8k"]["score"],
        "total_physical_seconds": sum(arm["physical_seconds"] for arm in arms.values()),
        "measured_serving_seconds": sum(
            repetition["duration"]
            for arm in arms.values()
            for repetition in arm["serving_repetitions"]
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
