"""Validate the fixed Wan all-step teacher-forced correctness matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
)


WAN_TIMESTEPS = (999, 970, 937, 899, 857, 807, 749, 681, 599, 499, 374, 214)
WAN_BLOCK_INDICES = tuple(range(40))
EXPECTED_RUNS = 2
EXPECTED_RECORDS_PER_RUN = 80
EXPECTED_CANDIDATE_HITS_PER_RUN = 160


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_tensor_metrics(
    metrics: dict[str, Any], *, context: str, exact: bool
) -> None:
    _require(metrics.get("finite") is True, f"{context}: non-finite tensor")
    for name in ("cosine_similarity", "mae", "max_abs"):
        _require(
            _finite_number(metrics.get(name)),
            f"{context}: missing finite {name}",
        )
    if exact:
        _require(
            metrics.get("exact_match") is True,
            f"{context}: not bitwise repeatable",
        )
        _require(
            float(metrics["mae"]) == 0.0 and float(metrics["max_abs"]) == 0.0,
            f"{context}: exact comparison has nonzero error",
        )
        return
    _require(
        metrics.get("within_tolerance") is True,
        f"{context}: outside atol/rtol",
    )
    _require(
        float(metrics["cosine_similarity"])
        >= MODEL_QUALIFICATION_THRESHOLDS["cosine_min"],
        f"{context}: cosine below minimum",
    )
    _require(
        float(metrics["mae"]) <= MODEL_QUALIFICATION_THRESHOLDS["mae_max"],
        f"{context}: MAE above maximum",
    )


def _validate_trajectory_comparison(
    comparison: dict[str, Any], *, context: str, exact: bool
) -> None:
    trajectory = comparison.get("trajectory_metrics")
    _require(isinstance(trajectory, dict), f"{context}: missing trajectory")
    _require(trajectory.get("num_steps") == 12, f"{context}: expected 12 steps")
    for name in ("timesteps_available", "timesteps_finite", "timesteps_match"):
        _require(trajectory.get(name) is True, f"{context}: {name} is not true")
    rows = trajectory.get("per_step_metrics")
    _require(isinstance(rows, list) and len(rows) == 12, f"{context}: bad steps")
    for step_index, (row, timestep) in enumerate(zip(rows, WAN_TIMESTEPS)):
        row_context = f"{context}/step-{step_index}"
        _require(row.get("step_index") == step_index, f"{row_context}: bad index")
        _require(
            float(row.get("reference_timestep")) == timestep
            and float(row.get("candidate_timestep")) == timestep,
            f"{row_context}: timestep mismatch",
        )
        _validate_tensor_metrics(row, context=row_context, exact=exact)
    output = comparison.get("output_metrics", {}).get("all_frames_metrics")
    _require(isinstance(output, dict), f"{context}: missing final-frame metrics")
    _validate_tensor_metrics(output, context=f"{context}/frames", exact=exact)


def _validate_pair_summary(
    summary: dict[str, Any], *, context: str, pairs: int, exact: bool
) -> None:
    if exact:
        _require(summary.get("available") is True, f"{context}: unavailable")
        _require(summary.get("num_runs") == EXPECTED_RUNS, f"{context}: bad runs")
        _require(summary.get("pairing") == "all-pairs", f"{context}: bad pairing")
    else:
        _require(
            summary.get("reference_num_runs") == EXPECTED_RUNS
            and summary.get("candidate_num_runs") == EXPECTED_RUNS,
            f"{context}: bad run counts",
        )
        _require(
            summary.get("pairing") == "cross-product",
            f"{context}: bad pairing",
        )
    comparisons = summary.get("comparisons")
    _require(
        summary.get("num_pairs") == pairs
        and isinstance(comparisons, list)
        and len(comparisons) == pairs,
        f"{context}: bad pair count",
    )
    expected_pair_ids = (
        {(0, 1)} if exact else {(0, 0), (0, 1), (1, 0), (1, 1)}
    )
    pair_ids = {
        (
            comparison.get("reference_run_index"),
            comparison.get("candidate_run_index"),
        )
        for comparison in comparisons
    }
    _require(pair_ids == expected_pair_ids, f"{context}: incomplete pair coverage")
    for pair_index, comparison in enumerate(comparisons):
        _validate_trajectory_comparison(
            comparison,
            context=f"{context}/pair-{pair_index}",
            exact=exact,
        )


def _parse_candidate_config(report: dict[str, Any], *, context: str) -> dict[str, Any]:
    value = report.get("server_kwargs", {}).get("candidate", {}).get(
        "attention_backend_config"
    )
    _require(isinstance(value, str), f"{context}: missing candidate config")
    try:
        config = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: malformed candidate config") from error
    _require(isinstance(config, dict), f"{context}: candidate config is not an object")
    return config


def _validate_provenance(
    report: dict[str, Any],
    *,
    context: str,
    expected_sglang_revision: str,
    expected_sglang_tree: str,
    expected_flashinfer_revision: str,
    expected_flashinfer_tree: str,
    expected_model_path: str,
    expected_device_name: str,
) -> None:
    source = report.get("source_identity", {})
    for name, revision, tree in (
        ("sglang", expected_sglang_revision, expected_sglang_tree),
        ("flashinfer", expected_flashinfer_revision, expected_flashinfer_tree),
    ):
        identity = source.get(name, {})
        _require(
            identity.get("git_revision") == revision,
            f"{context}: bad {name} revision",
        )
        _require(identity.get("git_tree") == tree, f"{context}: bad {name} tree")
        _require(identity.get("git_clean") is True, f"{context}: dirty {name} source")
    _require(
        report.get("model_path") == expected_model_path,
        f"{context}: bad model path",
    )
    device = report.get("device_identity", {})
    _require(device.get("name") == expected_device_name, f"{context}: bad GPU")
    _require(
        device.get("compute_capability") == [10, 0],
        f"{context}: bad compute capability",
    )


def _validate_report(
    report: dict[str, Any],
    *,
    step_index: int,
    timestep: int,
    expected_sglang_revision: str,
    expected_sglang_tree: str,
    expected_flashinfer_revision: str,
    expected_flashinfer_tree: str,
    expected_model_path: str,
    expected_device_name: str,
) -> set[tuple[int, int, bool]]:
    context = f"step-{step_index}/timestep-{timestep}"
    _require(report.get("schema_version") == 2, f"{context}: bad schema")
    _require(report.get("comparison_mode") == "correctness", f"{context}: wrong mode")
    _require(report.get("run_order") == "reference-first", f"{context}: wrong order")
    _require(report.get("warmup_runs") == 0, f"{context}: unexpected warmup")
    _require(report.get("seed") == 42, f"{context}: unexpected seed")
    _require(
        report.get("measure_runs") == EXPECTED_RUNS,
        f"{context}: bad measured runs",
    )
    _require(
        report.get("trajectory_capture_enabled") is True,
        f"{context}: trajectory off",
    )
    _require(
        report.get("qualification", {}).get("passed") is True,
        f"{context}: unqualified",
    )
    _validate_provenance(
        report,
        context=context,
        expected_sglang_revision=expected_sglang_revision,
        expected_sglang_tree=expected_sglang_tree,
        expected_flashinfer_revision=expected_flashinfer_revision,
        expected_flashinfer_tree=expected_flashinfer_tree,
        expected_model_path=expected_model_path,
        expected_device_name=expected_device_name,
    )
    sampling = report.get("sampling_kwargs", {})
    _require(
        sampling.get("num_inference_steps") == 12
        and sampling.get("width") == 640
        and sampling.get("height") == 384
        and sampling.get("num_frames") == 17,
        f"{context}: sampling contract mismatch",
    )
    _require(
        float(sampling.get("guidance_scale")) == 5.0,
        f"{context}: guidance scale mismatch",
    )
    _require(
        sampling.get("return_trajectory_latents") is True
        and sampling.get("return_trajectory_decoded") is False,
        f"{context}: trajectory mode mismatch",
    )
    _require(
        report.get("server_kwargs", {}).get("reference", {}).get("attention_backend")
        == "fa",
        f"{context}: reference is not FA4",
    )
    for variant in ("reference", "candidate"):
        server_kwargs = report.get("server_kwargs", {}).get(variant, {})
        _require(
            server_kwargs.get("enable_cfg_parallel") is False,
            f"{context}: {variant} did not use serial CFG",
        )
        _require(
            server_kwargs.get("num_gpus") == 1,
            f"{context}: {variant} did not use one GPU",
        )
    candidate_kwargs = report.get("server_kwargs", {}).get("candidate", {})
    _require(
        candidate_kwargs.get("attention_backend") == "wan_hybrid",
        f"{context}: bad candidate",
    )
    config = _parse_candidate_config(report, context=context)
    _require(
        config.get("wan_hybrid_layer_indices") == list(WAN_BLOCK_INDICES),
        f"{context}: bad blocks",
    )
    _require(
        config.get("wan_hybrid_teacher_forced_compare") is True,
        f"{context}: teacher forcing off",
    )
    _require(
        float(config.get("wan_hybrid_teacher_forced_timestep")) == timestep,
        f"{context}: bad target",
    )

    reference = report.get("reference_generation", {})
    candidate = report.get("candidate_generation", {})
    _require(
        reference.get("per_run_wan_hybrid_hit_count") == [0, 0],
        f"{context}: reference hits",
    )
    _require(
        candidate.get("per_run_wan_hybrid_hit_count")
        == [EXPECTED_CANDIDATE_HITS_PER_RUN] * EXPECTED_RUNS,
        f"{context}: candidate routed outside the target or missed repeats",
    )
    per_run = candidate.get("per_run_wan_hybrid_teacher_forced_blocks")
    _require(
        isinstance(per_run, list) and len(per_run) == EXPECTED_RUNS,
        f"{context}: missing runs",
    )
    expected_identities = {
        (block, cfg) for block in WAN_BLOCK_INDICES for cfg in (False, True)
    }
    logical_cells: set[tuple[int, int, bool]] = set()
    for run_index, records in enumerate(per_run):
        run_context = f"{context}/run-{run_index}"
        _require(
            isinstance(records, list) and len(records) == EXPECTED_RECORDS_PER_RUN,
            f"{run_context}: expected {EXPECTED_RECORDS_PER_RUN} records",
        )
        identities = {
            (record.get("block_index"), record.get("cfg_negative"))
            for record in records
        }
        _require(
            identities == expected_identities,
            f"{run_context}: block/CFG coverage mismatch",
        )
        for record in records:
            block_index = int(record["block_index"])
            cfg_negative = bool(record["cfg_negative"])
            record_context = (
                f"{run_context}/block-{block_index}/cfg-{int(cfg_negative)}"
            )
            _require(
                float(record.get("timestep")) == timestep,
                f"{record_context}: configured target mismatch",
            )
            _require(
                float(record.get("actual_timestep")) == timestep,
                f"{record_context}: actual target mismatch",
            )
            _require(
                record.get("denoising_step_index") == step_index,
                f"{record_context}: step mismatch",
            )
            for name in ("attention_output", "post_residual"):
                _validate_tensor_metrics(
                    record[name], context=f"{record_context}/{name}", exact=False
                )
                _validate_tensor_metrics(
                    record["candidate_repeatability"][name],
                    context=f"{record_context}/{name}-repeat",
                    exact=True,
                )
            logical_cells.add((step_index, block_index, cfg_negative))

    repeatability = report.get("repeatability", {})
    for name in ("reference", "candidate"):
        _validate_pair_summary(
            repeatability.get(name, {}),
            context=f"{context}/{name}-repeatability",
            pairs=1,
            exact=True,
        )
    _validate_pair_summary(
        report.get("cross_variant_metrics", {}),
        context=f"{context}/cross-variant",
        pairs=4,
        exact=False,
    )
    return logical_cells


def validate_teacher_forced_matrix(
    reports: list[dict[str, Any]],
    *,
    expected_sglang_revision: str,
    expected_sglang_tree: str,
    expected_flashinfer_revision: str,
    expected_flashinfer_tree: str,
    expected_model_path: str,
    expected_device_name: str = "NVIDIA B200",
) -> dict[str, Any]:
    _require(len(reports) == len(WAN_TIMESTEPS), "expected exactly 12 reports")
    by_timestep: dict[int, dict[str, Any]] = {}
    prompts = set()
    device_uuids = set()
    for report in reports:
        config = _parse_candidate_config(report, context="matrix")
        target = config.get("wan_hybrid_teacher_forced_timestep")
        _require(_finite_number(target), "matrix: missing finite target timestep")
        target_int = int(float(target))
        _require(
            float(target) == target_int,
            "matrix: target timestep must be integral",
        )
        _require(
            target_int not in by_timestep,
            f"matrix: duplicate target {target_int}",
        )
        by_timestep[target_int] = report
        prompts.add(report.get("prompt"))
        device_uuids.add(report.get("device_identity", {}).get("uuid"))
    _require(
        set(by_timestep) == set(WAN_TIMESTEPS),
        "matrix: target timestep coverage mismatch",
    )
    _require(
        len(prompts) == 1 and None not in prompts and "" not in prompts,
        "matrix: prompt mismatch",
    )
    _require(
        len(device_uuids) == 1 and None not in device_uuids,
        "matrix: physical GPU mismatch",
    )

    cells: set[tuple[int, int, bool]] = set()
    for step_index, timestep in enumerate(WAN_TIMESTEPS):
        cells.update(
            _validate_report(
                by_timestep[timestep],
                step_index=step_index,
                timestep=timestep,
                expected_sglang_revision=expected_sglang_revision,
                expected_sglang_tree=expected_sglang_tree,
                expected_flashinfer_revision=expected_flashinfer_revision,
                expected_flashinfer_tree=expected_flashinfer_tree,
                expected_model_path=expected_model_path,
                expected_device_name=expected_device_name,
            )
        )
    _require(len(cells) == 960, f"matrix: expected 960 logical cells, got {len(cells)}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "timesteps": list(WAN_TIMESTEPS),
        "blocks": list(WAN_BLOCK_INDICES),
        "cfg_sides": [False, True],
        "reports": 12,
        "measured_runs_per_report": EXPECTED_RUNS,
        "logical_cells": len(cells),
        "record_instances": len(cells) * EXPECTED_RUNS,
        "same_variant_pairs": {"reference": 12, "candidate": 12},
        "cross_variant_pairs": 48,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sglang-revision", required=True)
    parser.add_argument("--expected-sglang-tree", required=True)
    parser.add_argument("--expected-flashinfer-revision", required=True)
    parser.add_argument("--expected-flashinfer-tree", required=True)
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--expected-device-name", default="NVIDIA B200")
    args = parser.parse_args()

    paths = [Path(value).expanduser().resolve() for value in args.report]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = validate_teacher_forced_matrix(
        reports,
        expected_sglang_revision=args.expected_sglang_revision,
        expected_sglang_tree=args.expected_sglang_tree,
        expected_flashinfer_revision=args.expected_flashinfer_revision,
        expected_flashinfer_tree=args.expected_flashinfer_tree,
        expected_model_path=args.expected_model_path,
        expected_device_name=args.expected_device_name,
    )
    result["report_files"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in paths
    ]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
