#!/usr/bin/env python3
"""Fail-closed three-arm accuracy, E2E, identity, and artifact summary."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
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
EXPECTED_KERNEL_SYMBOL = "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4"


def load(path):
    return json.loads(path.read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_one(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one fresh JSONL row in {path}, found {len(lines)}")
    return json.loads(lines[0])


def validate_serving(path, expected_prompts):
    row = load_jsonl_one(path)
    expected = {
        "backend": "sglang",
        "dataset_name": "random",
        "max_concurrency": 64,
        "random_input_len": 4096,
        "random_output_len": 128,
        "random_range_ratio": 0.0,
        "completed": expected_prompts,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"{path} has {key}={row.get(key)!r}, expected {value!r}")
    if not isinstance(row.get("request_rate"), (int, float)) or not math.isinf(row["request_rate"]):
        raise RuntimeError(f"{path} request_rate is not infinity")
    for key in METRICS:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{path} has invalid metric {key}={value}")
    for key, expected_value in (("input_lens", 4096), ("output_lens", 128)):
        values = row.get(key)
        if not isinstance(values, list) or len(values) != expected_prompts:
            raise RuntimeError(f"{path} has wrong {key} count")
        if any(value != expected_value for value in values):
            raise RuntimeError(f"{path} has unexpected {key}")
    errors = row.get("errors")
    if not isinstance(errors, list) or len(errors) != expected_prompts or any(errors):
        raise RuntimeError(f"{path} contains failed requests: {errors}")
    return {"sha256": sha256_file(path), "metrics": row}


def require_file(path):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Required regular artifact missing: {path}")
    return path


def validate_shutdown(path):
    rows = {}
    for line in require_file(path).read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in rows:
            raise RuntimeError(f"Invalid server shutdown receipt: {path}")
        rows[key] = value
    if rows.get("signal") != "TERM" or rows.get("process_group_stopped") != "true":
        raise RuntimeError(f"Server did not stop cleanly: {rows}")
    if rows.get("exit_code") not in {"0", "143"}:
        raise RuntimeError(f"Unexpected server exit code: {rows}")
    return rows


def validate_gpu_receipt(path, runtime_contract):
    with require_file(path).open(newline="") as stream:
        rows = [[field.strip() for field in row] for row in csv.reader(stream)]
    expected = [
        [
            gpu["index"],
            gpu["name"],
            gpu["uuid"],
            gpu["compute_cap"],
            gpu["pci_bus_id"],
            gpu["memory_mib"],
        ]
        for gpu in runtime_contract["gpus"]
    ]
    if rows != expected:
        raise RuntimeError(f"Per-arm GPU identity differs from runtime contract: {rows}")
    return {"sha256": sha256_file(path), "rows": rows}


def summarize_arm(root, name, runtime_contract, input_contract):
    path = root / name
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"Arm artifact is not a fresh directory: {path}")
    complete = load(require_file(path / "COMPLETE"))
    expected_complete = {
        "status": "pass",
        "kind": "arm",
        "variant": name,
        "runtime_contract_sha256": sha256_file(root / "runtime-contract.json"),
        "slurm_job_id": runtime_contract["slurm_job_id"],
        "container_image": runtime_contract["container_image"],
        "sglang_commit": runtime_contract["sglang"]["commit"],
        "sglang_tree": runtime_contract["sglang"]["tree"],
        "flashinfer_commit": runtime_contract["flashinfer"]["commit"],
        "flashinfer_tree": runtime_contract["flashinfer"]["tree"],
        "flashinfer_wheel_sha256": runtime_contract["flashinfer"]["wheel_sha256"],
        "flashinfer_api_signature": runtime_contract["flashinfer"]["api_signature"],
    }
    if any(complete.get(key) != value for key, value in expected_complete.items()):
        raise RuntimeError(f"Arm {name} is not complete")
    common = (
        "environment.json",
        "physical-seconds.txt",
        "server-ready-seconds.txt",
        "server-shutdown.txt",
        "server-processes-before-stop.txt",
        "server-processes-after-stop.txt",
        "gpu-processes-after-stop.txt",
        "port-after-stop.json",
        "sglang-commit.txt",
        "sglang-tree.txt",
        "gpus.csv",
        "fixed-requests.json",
        "gsm8k.json",
        "gsm8k.html",
        "gsm8k-evidence.json",
        "serving-contract.json",
        "serving-warmup.jsonl",
        "serving-1.jsonl",
        "serving-2.jsonl",
        "serving-3.jsonl",
    )
    for filename in common:
        require_file(path / filename)
    sglang_commit = (path / "sglang-commit.txt").read_text().strip()
    sglang_tree = (path / "sglang-tree.txt").read_text().strip()
    if sglang_commit != runtime_contract["sglang"]["commit"] or sglang_tree != runtime_contract["sglang"]["tree"]:
        raise RuntimeError(f"Arm {name} SGLang source differs from runtime contract")

    warmup = validate_serving(path / "serving-warmup.jsonl", 32)
    repetition_receipts = [
        validate_serving(path / f"serving-{index}.jsonl", 256) for index in range(1, 4)
    ]
    repetitions = [receipt["metrics"] for receipt in repetition_receipts]
    medians = {key: statistics.median(row[key] for row in repetitions) for key in METRICS}
    gsm8k = load(path / "gsm8k.json")
    gsm8k_evidence = load(path / "gsm8k-evidence.json")
    if gsm8k_evidence.get("evaluated_examples") != 500 or gsm8k_evidence.get("score") != gsm8k.get("score"):
        raise RuntimeError(f"Arm {name} has invalid GSM8K evidence")
    serving_contract = load(path / "serving-contract.json")
    expected_serving_contract = {
        "model_path": input_contract["model_path"],
        "sharegpt_path": input_contract["sharegpt"]["path"],
        "sharegpt_sha256": input_contract["sharegpt"]["sha256"],
        "serving_selection_sha256": input_contract["sharegpt"][
            "serving_selection_sha256"
        ],
        "backend": "sglang",
        "dataset_name": "random",
        "warmup_prompts": 32,
        "measured_prompts_per_repetition": 256,
        "measured_repetitions": 3,
        "random_input_len": 4096,
        "random_output_len": 128,
        "random_range_ratio": 0.0,
        "request_rate": "infinity",
        "max_concurrency": 64,
        "seed": 20260825,
        "temperature": 0.0,
        "output_details": True,
    }
    if serving_contract != expected_serving_contract:
        raise RuntimeError(f"Arm {name} serving contract mismatch: {serving_contract}")

    if not (path / "server-processes-before-stop.txt").read_text().strip():
        raise RuntimeError(f"Arm {name} lacks a pre-shutdown process census")
    if (path / "server-processes-after-stop.txt").read_text():
        raise RuntimeError(f"Arm {name} retained server process-group members")
    if (path / "gpu-processes-after-stop.txt").read_text():
        raise RuntimeError(f"Arm {name} retained GPU compute processes")
    port_receipt = load(path / "port-after-stop.json")
    if port_receipt.get("connect_ex") == 0:
        raise RuntimeError(f"Arm {name} retained a listening server port")

    arm = {
        "status": "pass",
        "environment": load(path / "environment.json"),
        "source": {
            "sglang_commit": sglang_commit,
            "sglang_tree": sglang_tree,
            "gpus": validate_gpu_receipt(path / "gpus.csv", runtime_contract),
        },
        "physical_seconds": int((path / "physical-seconds.txt").read_text()),
        "server_ready_seconds": int((path / "server-ready-seconds.txt").read_text()),
        "server_shutdown": validate_shutdown(path / "server-shutdown.txt"),
        "port_after_shutdown": port_receipt,
        "gsm8k": gsm8k,
        "gsm8k_evidence": gsm8k_evidence,
        "serving_contract": serving_contract,
        "warmup": warmup,
        "serving_repetitions": repetitions,
        "serving_repetition_sha256": [receipt["sha256"] for receipt in repetition_receipts],
        "serving_medians": medians,
    }
    if name == "native":
        if (path / "route-evidence.json").exists() or (path / "source-patch-evidence.txt").exists():
            raise RuntimeError("Native arm contains experimental route evidence")
    else:
        arm["route_evidence"] = load(require_file(path / "route-evidence.json"))
        patch_lines = require_file(path / "source-patch-evidence.txt").read_text().splitlines()
        if len(patch_lines) != 4:
            raise RuntimeError(f"Arm {name} has wrong source-patch activation count")
        arm["source_patch_evidence_sha256"] = sha256_file(path / "source-patch-evidence.txt")
        arm["parity_vs_native"] = load(require_file(path / "parity-vs-native.json"))
        if arm["parity_vs_native"].get("exact_token_id_requests") != 32:
            raise RuntimeError(f"Arm {name} did not preserve fixed token ids")
    if name == "candidate":
        arm["cake_kernel_evidence"] = load(require_file(path / "cake-kernel-evidence.json"))
        if arm["cake_kernel_evidence"].get("rank_count") != 4:
            raise RuntimeError("Candidate lacks four-rank Cake kernel evidence")
        if arm["cake_kernel_evidence"].get("expected_kernel_symbol") != EXPECTED_KERNEL_SYMBOL:
            raise RuntimeError("Candidate trace checked the wrong kernel symbol")
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


def artifact_inventory(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Artifact tree contains symlink: {path}")
        if not path.is_file() or path == root / "summary.json" or path == root / "COMPLETE":
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def parse_slurm_time(value, timezone):
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm8k-min-score", type=float, default=0.90)
    parser.add_argument("--gsm8k-max-arm-delta", type=float, default=0.002)
    args = parser.parse_args()
    runtime_contract = load(require_file(args.result_root / "runtime-contract.json"))
    input_contract = load(require_file(args.result_root / "input-contract.json"))
    timing_start = load(require_file(args.result_root / "timing-start.json"))
    arms = {
        name: summarize_arm(args.result_root, name, runtime_contract, input_contract)
        for name in ("native", "explicit", "candidate")
    }
    gpu_hashes = {arm["source"]["gpus"]["sha256"] for arm in arms.values()}
    if len(gpu_hashes) != 1:
        raise RuntimeError("GPU identity changed between arms")
    scores = {name: arm["gsm8k"]["score"] for name, arm in arms.items()}
    if any(score < args.gsm8k_min_score for score in scores.values()):
        raise RuntimeError(f"GSM8K score is below {args.gsm8k_min_score}: {scores}")
    if max(scores.values()) - min(scores.values()) > args.gsm8k_max_arm_delta:
        raise RuntimeError(
            f"GSM8K arm delta exceeds {args.gsm8k_max_arm_delta}: {scores}"
        )
    inventory, inventory_sha = artifact_inventory(args.result_root)
    summary_complete = datetime.now().astimezone()
    scheduler_submit = parse_slurm_time(
        timing_start["scheduler_submit_time"], summary_complete.tzinfo
    )
    allocation_start = parse_slurm_time(
        timing_start["allocation_start_time"], summary_complete.tzinfo
    )
    if not scheduler_submit <= allocation_start <= summary_complete:
        raise RuntimeError(
            f"Invalid scheduler timing order: {scheduler_submit}, {allocation_start}, "
            f"{summary_complete}"
        )
    result = {
        "status": "pass",
        "model_repo": "meta-llama/Llama-3.1-70B-Instruct",
        "model_revision": "1605565b47bb9346c5515c34102e054115b4f98b",
        "seed": 20260825,
        "runtime_contract": runtime_contract,
        "input_contract": input_contract,
        "accuracy_gates": {
            "gsm8k_min_score": args.gsm8k_min_score,
            "gsm8k_max_arm_delta": args.gsm8k_max_arm_delta,
            "scores": scores,
        },
        "arms": arms,
        "fused_speedup_candidate_vs_explicit": ratios(arms["explicit"], arms["candidate"]),
        "net_candidate_vs_native": ratios(arms["native"], arms["candidate"]),
        "gsm8k_score_delta_candidate_minus_explicit": scores["candidate"] - scores["explicit"],
        "gsm8k_score_delta_candidate_minus_native": scores["candidate"] - scores["native"],
        "timing": {
            "scheduler_submit_time": scheduler_submit.isoformat(),
            "allocation_start_time": allocation_start.isoformat(),
            "summary_complete_time": summary_complete.isoformat(),
            "scheduler_queue_seconds": (allocation_start - scheduler_submit).total_seconds(),
            "allocation_physical_through_summary_seconds": (
                summary_complete - allocation_start
            ).total_seconds(),
            "scheduler_physical_turnaround_through_summary_seconds": (
                summary_complete - scheduler_submit
            ).total_seconds(),
            "three_arm_physical_seconds": sum(
                arm["physical_seconds"] for arm in arms.values()
            ),
            "input_weight_hash_seconds": input_contract[
                "runtime_weight_verification_seconds"
            ],
            "measured_repetition_count": 9,
            "measured_serving_seconds": sum(
                repetition["duration"]
                for arm in arms.values()
                for repetition in arm["serving_repetitions"]
            ),
        },
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": inventory_sha,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
