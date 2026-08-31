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
    EXACT_T4_ROUTE,
    FLASHINFER_COMMIT,
    HASHES,
    KL_SAMPLE_COUNT,
    KL_DIRECTION,
    KL_METRIC,
    KL_NORMALIZATION_ATOL,
    KL_POSITION_AGGREGATION,
    KL_SAMPLE_AGGREGATION,
    KL_TOKEN_ID_ORDER,
    KL_VOCAB_CHUNK_SIZE,
    MAX_KL_EXCLUSIVE,
    MIN_SCORE,
    MODEL_ID,
    MODEL_REVISION,
    MTP_PROBE_MAX_NEW_TOKENS,
    MTP_PROBE_PROMPT_INDEX,
    MTP_SPECULATIVE_EAGLE_TOPK,
    MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
    MTP_SPECULATIVE_NUM_STEPS,
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


def _server_hosts(binding: dict) -> dict[str, str]:
    hosts = binding.get("server_hosts")
    if not isinstance(hosts, dict) or set(hosts) != {"baseline", "candidate"}:
        raise ValueError("server_hosts must contain exactly baseline and candidate")
    if any(not isinstance(hosts[arm], str) or not hosts[arm].strip() for arm in hosts):
        raise ValueError("server_hosts values must be nonempty strings")
    return hosts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_vocab_size(model_path: Path) -> int:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"staged model config is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"staged model config is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"staged model config must be an object: {config_path}")
    vocab_size = config.get("vocab_size")
    if vocab_size is None and isinstance(config.get("text_config"), dict):
        vocab_size = config["text_config"].get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 1:
        raise ValueError("staged model config has no valid vocabulary size")
    return vocab_size


def _server_command(binding: dict, arm: str) -> list[str]:
    backend = "triton" if arm == "baseline" else "flashinfer"
    hosts = _server_hosts(binding)
    return [
        binding.get("python_executable", "python3"),
        "-m",
        "sglang.launch_server",
        "--model-path",
        binding["model_path"],
        "--tokenizer-path",
        binding["tokenizer_path"],
        "--host",
        hosts[arm],
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
        "--linear-attn-verify-backend",
        backend,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        str(MTP_SPECULATIVE_NUM_STEPS),
        "--speculative-eagle-topk",
        str(MTP_SPECULATIVE_EAGLE_TOPK),
        "--speculative-num-draft-tokens",
        str(MTP_SPECULATIVE_NUM_DRAFT_TOKENS),
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
    if not isinstance(binding.get("tokenizer_path"), str) or not Path(binding["tokenizer_path"]).is_dir():
        parser.error("tokenizer_path must name the staged tokenizer directory")
    if Path(binding["tokenizer_path"]).resolve() != Path(binding["model_path"]).resolve():
        parser.error("tokenizer_path must be the tokenizer inside the sealed model directory")
    if type(binding.get("vocab_size")) is not int or binding["vocab_size"] <= 1:
        parser.error("vocab_size must be an integer > 1")
    try:
        configured_vocab_size = _model_vocab_size(Path(binding["model_path"]))
    except ValueError as exc:
        parser.error(str(exc))
    if binding["vocab_size"] != configured_vocab_size:
        parser.error(
            f"vocab_size {binding['vocab_size']} != staged model config value "
            f"{configured_vocab_size}"
        )
    if binding.get("container_image") != CONTAINER_IMAGE:
        parser.error(f"container_image must be {CONTAINER_IMAGE}")
    if binding.get("compute_capability") != [10, 3]:
        parser.error("compute_capability must be [10, 3]")
    if set(binding.get("ports", {})) != {"baseline", "candidate"}:
        parser.error("ports must contain exactly baseline and candidate")
    try:
        server_hosts = _server_hosts(binding)
    except ValueError as exc:
        parser.error(str(exc))
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
                "base_url": f"http://{server_hosts[arm]}:{binding['ports'][arm]}",
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
            "metric": KL_METRIC,
            "direction": KL_DIRECTION,
            "position_aggregation": KL_POSITION_AGGREGATION,
            "sample_aggregation": KL_SAMPLE_AGGREGATION,
            "maximum_exclusive": MAX_KL_EXCLUSIVE,
            "full_vocabulary_per_scored_position": True,
            "top_k_truncation_allowed": False,
            "vocab_size": binding["vocab_size"],
            "token_id_order": KL_TOKEN_ID_ORDER,
            "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
            "normalization_atol": KL_NORMALIZATION_ATOL,
            "model_path": binding["model_path"],
            "tokenizer_path": binding["tokenizer_path"],
            "tokenizer_authority_sha256": HASHES["model_manifest_sha256"],
        },
        "mtp_probe": {
            "arms": ["baseline", "candidate"],
            "input_ids": artifacts["longbench_first48_ids"],
            "input_ids_sha256": HASHES["longbench_first48_ids_sha256"],
            "prompt_index": MTP_PROBE_PROMPT_INDEX,
            "requests_per_arm": 1,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": MTP_PROBE_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            "requested_speculative_algorithm": "NEXTN",
            "resolved_speculative_algorithm": "EAGLE",
            "speculative_num_steps": MTP_SPECULATIVE_NUM_STEPS,
            "speculative_eagle_topk": MTP_SPECULATIVE_EAGLE_TOPK,
            "speculative_num_draft_tokens": MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
            "server_info_must_match": True,
        },
        "routes": {
            "tp_size": TP_SIZE,
            "ranks": TP_RANKS,
            "candidate_exact_noncp_routes_on_every_rank": True,
            "candidate_exact_t4_route": EXACT_T4_ROUTE,
            "candidate_exact_t4_route_on_every_rank": True,
            "baseline_optimized_route_marker_count": 0,
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
