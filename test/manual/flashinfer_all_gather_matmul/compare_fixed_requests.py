#!/usr/bin/env python3
"""Compare fixed token sequences and logprobs against the native arm."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-exact", type=int, default=32)
    parser.add_argument("--max-logprob-delta", type=float, default=0.05)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    if len(reference) != len(candidate):
        raise RuntimeError("Fixed-request result counts differ")
    if len(reference) != args.required_exact:
        raise RuntimeError(
            f"Expected {args.required_exact} fixed requests, found {len(reference)}"
        )
    expected_indices = list(range(args.required_exact))
    if [row.get("index") for row in reference] != expected_indices:
        raise RuntimeError("Reference request indices are missing or out of order")
    if [row.get("index") for row in candidate] != expected_indices:
        raise RuntimeError("Candidate request indices are missing or out of order")

    exact = 0
    compared_logprobs = []
    rows = []
    for ref, got in zip(reference, candidate):
        if ref["prompt_sha256"] != got["prompt_sha256"]:
            raise RuntimeError(f"Prompt mismatch at index {ref['index']}")
        ref_ids, got_ids = ref["output_ids"], got["output_ids"]
        ref_logprobs, got_logprobs = ref["output_logprobs"], got["output_logprobs"]
        if not ref_ids or not got_ids:
            raise RuntimeError(f"Empty output token sequence at index {ref['index']}")
        if len(ref_ids) != len(ref_logprobs) or len(got_ids) != len(got_logprobs):
            raise RuntimeError(f"Token/logprob length mismatch at index {ref['index']}")
        if not all(math.isfinite(value) for value in ref_logprobs + got_logprobs):
            raise RuntimeError(f"Non-finite output logprob at index {ref['index']}")
        common = 0
        for ref_id, got_id in zip(ref_ids, got_ids):
            if ref_id != got_id:
                break
            common += 1
        if ref_ids == got_ids:
            exact += 1
        deltas = [
            abs(a - b)
            for a, b in zip(
                ref_logprobs[:common], got_logprobs[:common]
            )
        ]
        compared_logprobs.extend(deltas)
        rows.append(
            {
                "index": ref["index"],
                "exact_token_ids": ref_ids == got_ids,
                "reference_tokens": len(ref_ids),
                "candidate_tokens": len(got_ids),
                "common_prefix_tokens": common,
                "common_prefix_max_abs_logprob_delta": max(deltas, default=0.0),
            }
        )

    result = {
        "request_count": len(reference),
        "exact_token_id_requests": exact,
        "exact_token_id_fraction": exact / len(reference),
        "common_prefix_logprob_count": len(compared_logprobs),
        "common_prefix_max_abs_logprob_delta": max(compared_logprobs, default=0.0),
        "required_exact_token_id_requests": args.required_exact,
        "allowed_max_abs_logprob_delta": args.max_logprob_delta,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if exact != args.required_exact:
        raise RuntimeError(
            f"Token parity gate failed: exact={exact}, required={args.required_exact}"
        )
    maximum = result["common_prefix_max_abs_logprob_delta"]
    if maximum > args.max_logprob_delta:
        raise RuntimeError(
            f"Logprob parity gate failed: max={maximum}, allowed={args.max_logprob_delta}"
        )


if __name__ == "__main__":
    main()
