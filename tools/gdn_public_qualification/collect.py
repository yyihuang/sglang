"""Collect source-owned raw observations from an already running SGLang server.

This client deliberately does not start or stop a server.  The rendered plan
defines the two server configurations and ABBA order; this module gives each
measurement a small, auditable, one-purpose command.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from tools.gdn_public_qualification.contract import GSM8K_SHOTS, PROMPT_COUNT

INVALID_ANSWER = object()
ROUTE_RE = re.compile(r"Using (cake\.gdn_(prefill|decode)\.noncp\.[a-z0-9_.-]+)")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path} SHA256 {actual} != {expected}")


def _post(base_url: str, endpoint: str, payload: dict, timeout: float) -> object:
    response = requests.post(base_url.rstrip("/") + endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _flush(base_url: str, timeout: float) -> None:
    response = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=timeout)
    response.raise_for_status()


def _answer_value(text: str) -> object:
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    try:
        value = ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID_ANSWER
    return value if isinstance(value, (int, float)) and math.isfinite(value) else INVALID_ANSWER


def _gsm8k_example(row: dict, include_answer: bool) -> str:
    text = f"Question: {row['question']}\nAnswer:"
    return text + (f" {row['answer']}" if include_answer else "")


def collect_accuracy(args: argparse.Namespace) -> None:
    _require_hash(args.dataset, args.dataset_sha256)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if len(rows) != GSM8K_SHOTS + PROMPT_COUNT:
        raise ValueError(f"sealed GSM8K file must contain exactly 1319 rows, got {len(rows)}")
    shots = "".join(_gsm8k_example(row, True) + "\n\n" for row in rows[:GSM8K_SHOTS])

    def run(question_index: int) -> dict:
        source_row_index = GSM8K_SHOTS + question_index
        prompt = shots + _gsm8k_example(rows[source_row_index], False)
        result = _post(
            args.base_url,
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 512,
                },
            },
            args.timeout,
        )
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError(f"prompt {question_index} returned an invalid response")
        expected = _answer_value(rows[source_row_index]["answer"])
        observed = _answer_value(result["text"])
        return {
            "question_index": question_index,
            "source_row_index": source_row_index,
            "request_count": 1,
            "correct": observed is not INVALID_ANSWER and observed == expected,
        }

    _flush(args.base_url, args.timeout)
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        prompt_rows = list(executor.map(run, range(PROMPT_COUNT)))
    score = sum(row["correct"] for row in prompt_rows) / PROMPT_COUNT
    _write_json(args.output, {"arm": args.arm, "score": score, "prompts": prompt_rows})


def _load_input_ids(path: Path, expected_count: int) -> list[list[int]]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key in ("input_ids", "prompts", "records"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = [row["input_ids"] for row in value]
    if not (
        isinstance(value, list)
        and len(value) == expected_count
        and all(isinstance(row, list) and row and all(isinstance(token, int) for token in row) for row in value)
    ):
        raise ValueError(f"{path} must contain exactly {expected_count} non-empty token-ID rows")
    return value


def _as_results(value: object, expected: int) -> list[dict]:
    rows = value if isinstance(value, list) else [value]
    if len(rows) != expected or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"server returned {len(rows)} results, expected {expected}")
    return rows


def collect_kl_reference(args: argparse.Namespace) -> None:
    _require_hash(args.input_ids, args.input_ids_sha256)
    input_ids = _load_input_ids(args.input_ids, 48)
    _flush(args.base_url, args.timeout)
    results = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 512, "ignore_eos": True},
                "return_logprob": True,
                "return_text_in_logprobs": False,
            },
            args.timeout,
        ),
        48,
    )
    records = []
    for sample_index, result in enumerate(results):
        output_ids = result.get("output_ids")
        output_logprobs = result.get("meta_info", {}).get("output_token_logprobs")
        if not isinstance(output_ids, list) or not isinstance(output_logprobs, list):
            raise ValueError(f"KL reference sample {sample_index} has no token logprobs")
        records.append(
            {
                "sample_index": sample_index,
                "input_ids": input_ids[sample_index],
                "output_ids": output_ids,
                "baseline_logprobs": [row[0] for row in output_logprobs],
            }
        )
    _write_json(args.output, {"records": records})


def collect_kl_candidate(args: argparse.Namespace) -> None:
    reference = json.loads(args.reference.read_text())
    records = reference.get("records")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("KL reference must contain exactly 48 samples")
    joined = [row["input_ids"] + row["output_ids"] for row in records]
    _flush(args.base_url, args.timeout)
    results = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": joined,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": 0,
            },
            args.timeout,
        ),
        48,
    )
    output = []
    for sample_index, (reference_row, result) in enumerate(zip(records, results)):
        token_count = len(reference_row["output_ids"])
        input_logprobs = result.get("meta_info", {}).get("input_token_logprobs")
        if not isinstance(input_logprobs, list) or len(input_logprobs) < token_count:
            raise ValueError(f"KL candidate sample {sample_index} has incomplete logprobs")
        output.append(
            {
                "sample_index": sample_index,
                "baseline_logprobs": reference_row["baseline_logprobs"],
                "candidate_logprobs": [row[0] for row in input_logprobs[-token_count:]],
            }
        )
    _write_json(args.output, {"records": output})


def collect_performance(args: argparse.Namespace) -> None:
    _require_hash(args.input_ids, args.input_ids_sha256)
    input_ids = _load_input_ids(args.input_ids, args.prompt_count)
    _flush(args.base_url, args.timeout)
    start = time.perf_counter()
    results = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 512, "ignore_eos": True},
            },
            args.timeout,
        ),
        args.prompt_count,
    )
    elapsed = time.perf_counter() - start
    output_tokens = sum(len(row.get("output_ids", [])) for row in results)
    if output_tokens != args.prompt_count * 512:
        raise ValueError(f"performance run returned {output_tokens} output tokens, expected {args.prompt_count * 512}")
    _write_json(
        args.output,
        {
            "arm": args.arm,
            "workload_id": args.workload_id,
            "input_ids_sha256": args.input_ids_sha256,
            "throughput_tokens_per_second": output_tokens / elapsed,
            "measured_runtime_seconds": elapsed,
            "output_tokens": output_tokens,
        },
    )


def collect_routes(args: argparse.Namespace) -> None:
    rank_rows = []
    for rank_log in args.rank_log:
        rank_text, path_text = rank_log.split(":", 1)
        rank = int(rank_text)
        path = Path(path_text)
        text = path.read_text(errors="replace")
        matches = list(ROUTE_RE.finditer(text))
        prefill = sorted({match.group(1) for match in matches if match.group(2) == "prefill"})
        decode = sorted({match.group(1) for match in matches if match.group(2) == "decode"})
        rank_rows.append(
            {
                "rank": rank,
                "prefill_routes": prefill,
                "decode_routes": decode,
                "cake_route_count": len(matches),
                "fallback_count": len(re.findall(r"fall(?:ing)? back", text, flags=re.IGNORECASE)),
                "route_error_count": len(re.findall(r"cake\.gdn_.*(?:error|failed)", text, flags=re.IGNORECASE)),
                "log_sha256": _sha256(path),
            }
        )
    rank_rows.sort(key=lambda row: row["rank"])
    _write_json(args.output, {"arm": args.arm, "ranks": rank_rows})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def server_parser(name: str, function):
        child = subparsers.add_parser(name)
        child.add_argument("--base-url", required=True)
        child.add_argument("--timeout", type=float, default=7200)
        child.add_argument("--output", type=Path, required=True)
        child.set_defaults(function=function)
        return child

    accuracy = server_parser("accuracy", collect_accuracy)
    accuracy.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    accuracy.add_argument("--dataset", type=Path, required=True)
    accuracy.add_argument("--dataset-sha256", required=True)
    accuracy.add_argument("--parallel", type=int, default=128)

    kl_reference = server_parser("kl-reference", collect_kl_reference)
    kl_reference.add_argument("--input-ids", type=Path, required=True)
    kl_reference.add_argument("--input-ids-sha256", required=True)

    kl_candidate = server_parser("kl-candidate", collect_kl_candidate)
    kl_candidate.add_argument("--reference", type=Path, required=True)

    performance = server_parser("performance", collect_performance)
    performance.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    performance.add_argument("--workload-id", required=True)
    performance.add_argument("--input-ids", type=Path, required=True)
    performance.add_argument("--input-ids-sha256", required=True)
    performance.add_argument("--prompt-count", type=int, choices=(32, 48), required=True)

    routes = subparsers.add_parser("routes")
    routes.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    routes.add_argument("--rank-log", action="append", required=True, help="RANK:PATH; pass exactly four")
    routes.add_argument("--output", type=Path, required=True)
    routes.set_defaults(function=collect_routes)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
