"""Client-side correctness and workload driver for the GB300 CAKE E2E test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


PERF_PROMPT = (
    "Explain why the union of CUDA kernel execution intervals excludes host "
    "launch gaps when comparing two GPU attention backends. "
) * 8


def _generate(
    port: int,
    text: str,
    max_new_tokens: int,
    *,
    stop: list[str] | None = None,
    ignore_eos: bool = False,
) -> dict:
    sampling_params: dict = {
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
    }
    if stop is not None:
        sampling_params["stop"] = stop
        sampling_params["ignore_eos"] = False
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"text": text, "sampling_params": sampling_params},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _check(name: str, passed: bool, response: dict, **diagnostic) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "text": response["text"],
        "output_ids": response.get("output_ids", []),
        **diagnostic,
    }


def _correctness(port: int) -> dict:
    checks = []

    capital = _generate(
        port,
        "Q: What is the capital of France?\nA:",
        64,
    )
    checks.append(_check("capital_france", "paris" in capital["text"].lower(), capital))

    math = _generate(
        port,
        "Q: What is 17 multiplied by 23? Reply with just the number.\nA:",
        64,
    )
    checks.append(_check("basic_math", "391" in math["text"], math))

    color = _generate(
        port,
        "Q: The three primary colors are red, blue, and ___. Fill in the blank.\nA:",
        64,
    )
    checks.append(_check("color_completion", "yellow" in color["text"].lower(), color))

    ascii_response = _generate(
        port,
        "Write a single sentence about a sunny day in the park.",
        128,
    )
    ascii_text = ascii_response["text"]
    printable = sum(
        1 for char in ascii_text if 32 <= ord(char) < 127 or char in "\n\t"
    )
    ascii_ratio = printable / max(len(ascii_text), 1)
    checks.append(
        _check(
            "ascii_ratio",
            ascii_ratio > 0.85,
            ascii_response,
            printable_ascii_ratio=ascii_ratio,
        )
    )

    repetition = _generate(port, "Briefly explain what gravity is.", 128)
    repetition_text = repetition["text"]
    windows = [
        repetition_text[i : i + 5]
        for i in range(max(0, len(repetition_text) - 5))
    ]
    most_common = max((windows.count(window) for window in set(windows)), default=0)
    repetition_ratio = most_common / max(len(windows), 1)
    checks.append(
        _check(
            "no_repetition_blowup",
            len(repetition_text) < 50 or repetition_ratio < 0.25,
            repetition,
            top_5gram_ratio=repetition_ratio,
        )
    )

    deterministic_prompt = "Q: What is the capital of France? Reply in one word.\nA:"
    deterministic_a = _generate(port, deterministic_prompt, 64, stop=["\n"])
    deterministic_b = _generate(port, deterministic_prompt, 64, stop=["\n"])
    checks.append(
        _check(
            "determinism_temperature_zero",
            deterministic_a["text"].strip() == deterministic_b["text"].strip(),
            deterministic_a,
            second_text=deterministic_b["text"],
            second_output_ids=deterministic_b.get("output_ids", []),
        )
    )

    one_token = _generate(
        port,
        "Q: What is the capital of France? Just one word.\nA:",
        1,
    )
    checks.append(_check("max_token_one", bool(one_token["text"]), one_token))

    return {
        "all_passed": all(row["passed"] for row in checks),
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
        "checks": checks,
    }


def _perf(port: int) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(
            executor.map(
                lambda _: _generate(port, PERF_PROMPT, 64, ignore_eos=True),
                range(8),
            )
        )
    return {
        "requests": len(rows),
        "requested_output_tokens": 8 * 64,
        "observed_completion_tokens": sum(
            int(row.get("meta_info", {}).get("completion_tokens", 64)) for row in rows
        ),
        "rows": rows,
    }


def _compare(
    baseline_correctness: Path,
    candidate_correctness: Path,
    baseline_perf: Path,
    candidate_perf: Path,
) -> dict:
    baseline_checks = json.loads(baseline_correctness.read_text())
    candidate_checks = json.loads(candidate_correctness.read_text())
    baseline_rows = json.loads(baseline_perf.read_text())["rows"]
    candidate_rows = json.loads(candidate_perf.read_text())["rows"]
    exact_rows = [
        baseline["output_ids"] == candidate["output_ids"]
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True)
    ]
    exact_tokens = sum(
        len(baseline["output_ids"])
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True)
        if baseline["output_ids"] == candidate["output_ids"]
    )
    passed = (
        baseline_checks["all_passed"]
        and candidate_checks["all_passed"]
        and all(exact_rows)
    )
    return {
        "all_passed": passed,
        "baseline_sanity": {
            "passed": baseline_checks["passed"],
            "total": baseline_checks["total"],
        },
        "cake_sanity": {
            "passed": candidate_checks["passed"],
            "total": candidate_checks["total"],
        },
        "perf_workload_exact_rows": sum(exact_rows),
        "perf_workload_total_rows": len(exact_rows),
        "perf_workload_exact_output_tokens": exact_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("warmup", "correctness", "perf", "compare"))
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-correctness", type=Path)
    parser.add_argument("--candidate-correctness", type=Path)
    parser.add_argument("--baseline-perf", type=Path)
    parser.add_argument("--candidate-perf", type=Path)
    args = parser.parse_args()

    if args.mode == "warmup":
        payload = _generate(args.port, PERF_PROMPT, 8, ignore_eos=True)
    elif args.mode == "correctness":
        payload = _correctness(args.port)
    elif args.mode == "perf":
        payload = _perf(args.port)
    else:
        payload = _compare(
            args.baseline_correctness,
            args.candidate_correctness,
            args.baseline_perf,
            args.candidate_perf,
        )

    print("SGLANG_DSV4_E2E_CLIENT=" + json.dumps(payload, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.mode in ("correctness", "compare") and not payload["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
