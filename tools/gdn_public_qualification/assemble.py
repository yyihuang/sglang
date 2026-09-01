#!/usr/bin/env python3
"""Assemble one canonical v2 GDN qualification result from collector outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    KL_DIRECTION,
    KL_METRIC,
    KL_POSITION_AGGREGATION,
    KL_SAMPLE_AGGREGATION,
    PLAN_SCHEMA,
    ROUTE_ARTIFACT_SCHEMA,
    SCHEMA,
    WORKLOADS,
    QualificationError,
    load_campaign_plan,
    load_strict_json,
    recompute_kl_summary,
    validate_route_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} is not a regular file: {path}")
    try:
        value = load_strict_json(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise QualificationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _evidence_spec(path: Path, evidence_root: Path, label: str) -> dict[str, str]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise QualificationError(f"{label} is not a regular file: {path}")
    if not resolved.is_relative_to(evidence_root):
        raise QualificationError(f"{label} must be below the result evidence root")
    return {
        "path": str(resolved.relative_to(evidence_root)),
        "sha256": _sha256(resolved),
    }


def _iso8601(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000, tz=timezone.utc
    ).isoformat()


def _arm_output(path: Path, arm: str, label: str) -> dict[str, object]:
    value = _load_object(path, f"{arm} {label}")
    if value.get("arm") != arm:
        raise QualificationError(f"{arm} {label} arm identity differs")
    return value


def _accuracy_output(path: Path, arm: str, campaign_id: str) -> dict[str, object]:
    value = _arm_output(path, arm, "accuracy output")
    if set(value) != {
        "arm",
        "campaign_id",
        "plan_sha256",
        "dataset_sha256",
        "prompt_ids_sha256",
        "request_payload",
        "sampling_params",
        "server_config",
        "model_identity",
        "request_ledger",
        "server_request_evidence",
        "score",
        "prompts",
    }:
        raise QualificationError(f"{arm} accuracy output fields differ")
    if value.get("campaign_id") != campaign_id or value.get("plan_sha256") != campaign_id:
        raise QualificationError(f"{arm} accuracy output campaign identity differs")
    return value


def _mtp_output(path: Path, arm: str) -> dict[str, object]:
    value = _arm_output(path, arm, "MTP probe output")
    if set(value) != {
        "arm",
        "input_ids_sha256",
        "prompt_index",
        "request_count",
        "sampling_params",
        "server_config",
        "output_ids",
        "output_ids_sha256",
        "measured_runtime_seconds",
    }:
        raise QualificationError(f"{arm} MTP probe output fields differ")
    return value


def _route_output(path: Path, arm: str) -> list[object]:
    value = _arm_output(path, arm, "route output")
    if set(value) != {"arm", "ranks"} or not isinstance(value.get("ranks"), list):
        raise QualificationError(f"{arm} route output fields differ")
    return value["ranks"]


def _performance_outputs(directory: Path) -> list[dict[str, object]]:
    if directory.is_symlink() or not directory.is_dir():
        raise QualificationError(f"performance directory is invalid: {directory}")
    workloads = []
    for workload_id, input_ids_sha256 in WORKLOADS.items():
        observations = []
        for sequence_index, arm in enumerate(ABBA_ORDER):
            path = directory / (
                f"performance.{workload_id}.{sequence_index:02d}.{arm}.json"
            )
            value = _arm_output(path, arm, "performance output")
            if set(value) != {
                "arm",
                "workload_id",
                "input_ids_sha256",
                "throughput_tokens_per_second",
                "measured_runtime_seconds",
                "output_tokens",
            }:
                raise QualificationError(
                    f"performance output fields differ at {workload_id} "
                    f"observation {sequence_index}"
                )
            if (
                value.get("workload_id") != workload_id
                or value.get("input_ids_sha256") != input_ids_sha256
            ):
                raise QualificationError(
                    f"performance identity differs at {workload_id} "
                    f"observation {sequence_index}"
                )
            observations.append({"sequence_index": sequence_index, **value})
        workloads.append(
            {
                "workload_id": workload_id,
                "input_ids_sha256": input_ids_sha256,
                "observations": observations,
            }
        )
    return workloads


def assemble(args: argparse.Namespace) -> dict[str, object]:
    output = args.output
    if not output.is_absolute() or output != output.resolve():
        raise QualificationError("result output must be normalized and absolute")
    if output.exists() or not output.parent.is_dir():
        raise QualificationError("result output must be fresh in an existing directory")
    evidence_root = output.parent.resolve()

    for name in ("physical_start_ns", "measured_start_ns", "finish_ns"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise QualificationError(f"{name} must be a positive integer")
    if not args.physical_start_ns < args.measured_start_ns < args.finish_ns:
        raise QualificationError("campaign timestamps must be strictly ordered")

    for path, label in (
        (args.accuracy_baseline, "baseline accuracy output"),
        (args.accuracy_candidate, "candidate accuracy output"),
        (args.mtp_baseline, "baseline MTP probe output"),
        (args.mtp_candidate, "candidate MTP probe output"),
        (args.routes_baseline, "baseline route output"),
        (args.routes_candidate, "candidate route output"),
        (args.kl_baseline_manifest, "baseline KL manifest"),
        (args.kl_candidate_manifest, "candidate KL manifest"),
    ):
        _evidence_spec(path, evidence_root, label)
    try:
        performance_dir = args.performance_dir.resolve(strict=True)
    except OSError as exc:
        raise QualificationError("performance directory is missing") from exc
    if (
        args.performance_dir.is_symlink()
        or not performance_dir.is_dir()
        or not performance_dir.is_relative_to(evidence_root)
    ):
        raise QualificationError(
            "performance directory must be a regular directory below the result evidence root"
        )

    plan_spec = _evidence_spec(args.plan, evidence_root, "campaign plan")
    campaign_id = plan_spec["sha256"]
    plan = load_campaign_plan(plan_spec, evidence_root, campaign_id)
    if plan.get("schema") != PLAN_SCHEMA:
        raise QualificationError("campaign plan schema differs")
    provenance = plan.get("provenance")
    bindings = plan.get("bindings")
    if not isinstance(provenance, Mapping) or not isinstance(bindings, Mapping):
        raise QualificationError("campaign plan identity is incomplete")

    artifacts = bindings.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QualificationError("campaign plan artifact bindings are required")
    dataset_spec = _evidence_spec(
        Path(str(artifacts.get("gsm8k_dataset"))), evidence_root, "GSM8K dataset"
    )
    prompt_ids_spec = _evidence_spec(
        Path(str(artifacts.get("gsm8k_prompt_ids"))),
        evidence_root,
        "GSM8K prompt IDs",
    )

    route_binding = bindings.get("route_artifact")
    if not isinstance(route_binding, Mapping):
        raise QualificationError("campaign plan route artifact binding is required")
    route_path = Path(str(route_binding.get("path")))
    route_spec = _evidence_spec(route_path, evidence_root, "route artifact")
    if route_spec.get("sha256") != route_binding.get("sha256"):
        raise QualificationError("route artifact hash differs from the campaign plan")
    expected_routes = validate_route_artifact(
        _load_object(route_path, "route artifact")
    )
    plan_routes = plan.get("routes")
    if (
        not isinstance(plan_routes, Mapping)
        or plan_routes.get("artifact")
        != {"schema": ROUTE_ARTIFACT_SCHEMA, "sha256": route_spec["sha256"]}
        or plan_routes.get("expected_candidate_routes") != expected_routes
    ):
        raise QualificationError("route artifact content differs from the campaign plan")

    accuracy_arms = {
        "baseline": _accuracy_output(
            args.accuracy_baseline, "baseline", campaign_id
        ),
        "candidate": _accuracy_output(
            args.accuracy_candidate, "candidate", campaign_id
        ),
    }
    mtp_arms = {
        "baseline": _mtp_output(args.mtp_baseline, "baseline"),
        "candidate": _mtp_output(args.mtp_candidate, "candidate"),
    }
    route_arms = {
        "baseline": _route_output(args.routes_baseline, "baseline"),
        "candidate": _route_output(args.routes_candidate, "candidate"),
    }
    kl_specs = {
        "baseline_manifest": _evidence_spec(
            args.kl_baseline_manifest, evidence_root, "baseline KL manifest"
        ),
        "candidate_manifest": _evidence_spec(
            args.kl_candidate_manifest, evidence_root, "candidate KL manifest"
        ),
    }
    kl_summary = recompute_kl_summary(kl_specs, evidence_root)

    result = {
        "schema": SCHEMA,
        "provenance": dict(provenance),
        "campaign": {
            "started_at": _iso8601(args.physical_start_ns),
            "finished_at": _iso8601(args.finish_ns),
            "physical_turnaround_seconds": (
                args.finish_ns - args.physical_start_ns
            )
            / 1_000_000_000,
            "measured_runtime_seconds": (
                args.finish_ns - args.measured_start_ns
            )
            / 1_000_000_000,
        },
        "accuracy": {
            "campaign_id": campaign_id,
            "plan": plan_spec,
            "dataset": dataset_spec,
            "prompt_ids": prompt_ids_spec,
            "arms": accuracy_arms,
            "kl": {
                "metric": KL_METRIC,
                "direction": KL_DIRECTION,
                "position_aggregation": KL_POSITION_AGGREGATION,
                "sample_aggregation": KL_SAMPLE_AGGREGATION,
                **kl_summary,
                **kl_specs,
            },
        },
        "mtp_probe": {"arms": mtp_arms},
        "routes": {
            "artifact": route_spec,
            "expected_candidate_routes": expected_routes,
            "arms": route_arms,
        },
        "performance": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "workloads": _performance_outputs(performance_dir),
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--accuracy-baseline", type=Path, required=True)
    parser.add_argument("--accuracy-candidate", type=Path, required=True)
    parser.add_argument("--mtp-baseline", type=Path, required=True)
    parser.add_argument("--mtp-candidate", type=Path, required=True)
    parser.add_argument("--routes-baseline", type=Path, required=True)
    parser.add_argument("--routes-candidate", type=Path, required=True)
    parser.add_argument("--kl-baseline-manifest", type=Path, required=True)
    parser.add_argument("--kl-candidate-manifest", type=Path, required=True)
    parser.add_argument("--performance-dir", type=Path, required=True)
    parser.add_argument("--physical-start-ns", type=int, required=True)
    parser.add_argument("--measured-start-ns", type=int, required=True)
    parser.add_argument("--finish-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = assemble(args)
        with args.output.open("x") as handle:
            json.dump(result, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    except QualificationError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": SCHEMA,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
