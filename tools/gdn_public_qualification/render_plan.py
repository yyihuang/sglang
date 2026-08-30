"""Verify private staged inputs and render the immutable qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONTAINER_IMAGE,
    FLASHINFER_COMMIT,
    HASHES,
    KL_SAMPLE_COUNT,
    MAX_KL_EXCLUSIVE,
    MIN_SCORE,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATIONS_PER_ARM_PER_WORKLOAD,
    PLAN_SCHEMA,
    PROMPT_COUNT,
    SGLANG_INTEGRATION_COMMIT,
    TP_RANKS,
    TP_SIZE,
    WORKLOADS,
    expected_provenance,
)

ARTIFACT_HASH_KEYS = {
    "flashinfer_bundle": "flashinfer_bundle_sha256",
    "input_delivery_manifest": "input_delivery_manifest_sha256",
    "source_authority": "source_authority_sha256",
    "model_manifest": "model_manifest_sha256",
    "model_stage_receipt": "model_stage_receipt_sha256",
    "gsm8k_dataset": "gsm8k_dataset_sha256",
    "gsm8k_manifest": "gsm8k_manifest_sha256",
    "gsm8k_prompt_ids": "gsm8k_prompt_ids_sha256",
    "longbench_dataset": "longbench_dataset_sha256",
    "longbench_manifest": "longbench_manifest_sha256",
    "longbench_first32_ids": "longbench_first32_ids_sha256",
    "longbench_first48_ids": "longbench_first48_ids_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _server_command(binding: dict, arm: str) -> list[str]:
    backend = "triton" if arm == "baseline" else "flashinfer"
    return [
        binding.get("python_executable", "python3"),
        "-m",
        "sglang.launch_server",
        "--model-path",
        binding["model_path"],
        "--host",
        binding.get("server_host", "127.0.0.1"),
        "--port",
        str(binding["ports"][arm]),
        "--tp-size",
        "4",
        "--trust-remote-code",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--linear-attn-decode-backend",
        backend,
        "--linear-attn-prefill-backend",
        backend,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bindings", type=Path, help="private, uncommitted staged-path JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.bindings.read_text())

    repo_root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if not isinstance(binding.get("model_path"), str) or not Path(binding["model_path"]).is_dir():
        parser.error("model_path must name the staged model directory")
    if binding.get("container_image") != CONTAINER_IMAGE:
        parser.error(f"container_image must be {CONTAINER_IMAGE}")
    if binding.get("compute_capability") != [10, 3]:
        parser.error("compute_capability must be [10, 3]")
    if set(binding.get("ports", {})) != {"baseline", "candidate"}:
        parser.error("ports must contain exactly baseline and candidate")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_HASH_KEYS):
        parser.error(f"artifacts must contain exactly {sorted(ARTIFACT_HASH_KEYS)}")
    for artifact, hash_key in ARTIFACT_HASH_KEYS.items():
        path = Path(artifacts[artifact])
        if not path.is_file():
            parser.error(f"missing staged artifact {artifact}: {path}")
        actual = _sha256(path)
        if actual != HASHES[hash_key]:
            parser.error(f"{artifact} SHA256 {actual} != {HASHES[hash_key]}")

    provenance = expected_provenance()
    provenance.update(
        {
            "qualification_commit": head,
            "compute_capability": [10, 3],
            "gpu_name": binding.get("gpu_name", "NVIDIA GB300"),
            "cuda_version": binding["cuda_version"],
            "tp_size": TP_SIZE,
            "tp_ranks": TP_RANKS,
        }
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embargoed_until_arch_validation": True,
        "provenance": provenance,
        "bindings": binding,
        "servers": {
            arm: {
                "command": _server_command(binding, arm),
                "base_url": f"http://{binding.get('server_host', '127.0.0.1')}:{binding['ports'][arm]}",
                "pythonpath_prepend": binding["flashinfer_python_path"] if arm == "candidate" else None,
                "rank_logs": binding["rank_logs"][arm],
            }
            for arm in ("baseline", "candidate")
        },
        "accuracy": {
            "dataset": artifacts["gsm8k_dataset"],
            "prompt_ids": artifacts["gsm8k_prompt_ids"],
            "prompt_count": PROMPT_COUNT,
            "shots": 5,
            "requests_per_prompt_per_arm": 1,
            "minimum_score": MIN_SCORE,
            "candidate_no_drop": True,
        },
        "kl": {
            "input_ids": artifacts["longbench_first48_ids"],
            "sample_count": KL_SAMPLE_COUNT,
            "maximum_exclusive": MAX_KL_EXCLUSIVE,
        },
        "routes": {
            "tp_size": TP_SIZE,
            "ranks": TP_RANKS,
            "candidate_exact_noncp_routes_on_every_rank": True,
            "baseline_cake_route_count": 0,
        },
        "performance": {
            "workloads": [
                {
                    "workload_id": workload_id,
                    "input_ids": artifacts[f"longbench_first{32 if workload_id.endswith('32') else 48}_ids"],
                    "input_ids_sha256": digest,
                }
                for workload_id, digest in WORKLOADS.items()
            ],
            "order": ABBA_ORDER,
            "observations_per_arm_per_workload": OBSERVATIONS_PER_ARM_PER_WORKLOAD,
            "max_new_tokens": 512,
            "ignore_eos": True,
            "metric": "output_tokens_per_second",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "aggregate_geomean_must_exceed": 1.0,
            "aggregate_lower_95_must_exceed": 1.0,
            "resolved_regression_definition": "per-workload upper 95% bootstrap bound < 1",
        },
        "identity_checks": {
            "sglang_integration_commit": SGLANG_INTEGRATION_COMMIT,
            "flashinfer_commit": FLASHINFER_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
    }
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote verified private plan to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
