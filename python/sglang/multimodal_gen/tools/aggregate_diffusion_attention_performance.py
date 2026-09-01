"""Aggregate counterbalanced diffusion-attention performance reports.

The two inputs must be performance-mode reports produced with opposite run
orders from an otherwise identical source, configuration, runtime, and CUDA
device. The output fails closed unless both order-specific wall-clock speedups
reach the fixed qualification floor and every candidate run used the selected
backend.
"""

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


_CONFIGURATION_FIELDS = (
    "model_path",
    "prompt",
    "seed",
    "warmup_runs",
    "measure_runs",
    "server_kwargs",
    "backend_overrides",
    "sampling_kwargs",
)
_IDENTITY_FIELDS = (
    "source_identity",
    "device_identity",
    "runtime_provenance",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failure(report: str, field: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"report": report, "field": field, "reason": reason, **details}


def _validate_report(
    report: Any,
    *,
    label: str,
    expected_order: str,
) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return [
            _failure(
                label,
                "$",
                "report_is_not_an_object",
                actual_type=type(report).__name__,
            )
        ]

    failures: list[dict[str, Any]] = []
    if report.get("schema_version") != 2:
        failures.append(
            _failure(
                label,
                "schema_version",
                "unsupported_schema_version",
                expected=2,
                actual=report.get("schema_version"),
            )
        )
    if report.get("run_order") != expected_order:
        failures.append(
            _failure(
                label,
                "run_order",
                "unexpected_value",
                expected=expected_order,
                actual=report.get("run_order"),
            )
        )
    if report.get("comparison_mode") != "performance":
        failures.append(
            _failure(
                label,
                "comparison_mode",
                "performance_mode_required",
                actual=report.get("comparison_mode"),
            )
        )
    if report.get("trajectory_capture_enabled") is not False:
        failures.append(
            _failure(
                label,
                "trajectory_capture_enabled",
                "trajectory_capture_must_be_explicitly_disabled",
                actual=report.get("trajectory_capture_enabled"),
            )
        )
    for correctness_only_field in ("cross_variant_metrics", "repeatability"):
        if correctness_only_field in report:
            failures.append(
                _failure(
                    label,
                    correctness_only_field,
                    "correctness_payload_present_in_performance_report",
                )
            )

    for field in (*_CONFIGURATION_FIELDS, *_IDENTITY_FIELDS):
        value = report.get(field)
        if value is None or (isinstance(value, dict) and not value):
            failures.append(_failure(label, field, "missing_or_empty"))

    device_identity = report.get("device_identity")
    if isinstance(device_identity, dict) and not device_identity.get("uuid"):
        failures.append(
            _failure(
                label,
                "device_identity.uuid",
                "physical_device_uuid_required",
            )
        )

    source_identity = report.get("source_identity")
    if isinstance(source_identity, dict):
        for component in ("sglang", "flashinfer"):
            component_identity = source_identity.get(component)
            if not isinstance(component_identity, dict):
                failures.append(
                    _failure(
                        label,
                        f"source_identity.{component}",
                        "missing_or_empty",
                    )
                )
                continue
            for identity_field in (
                "git_revision",
                "git_tree",
                "git_status_sha256",
                "module_file_sha256",
            ):
                if not component_identity.get(identity_field):
                    failures.append(
                        _failure(
                            label,
                            f"source_identity.{component}.{identity_field}",
                            "missing_or_empty",
                        )
                    )
            if component_identity.get("git_clean") is not True:
                failures.append(
                    _failure(
                        label,
                        f"source_identity.{component}.git_clean",
                        "clean_source_tree_required",
                        actual=component_identity.get("git_clean"),
                    )
                )

    sampling_kwargs = report.get("sampling_kwargs")
    if isinstance(sampling_kwargs, dict) and sampling_kwargs.get(
        "save_output"
    ) is not False:
        failures.append(
            _failure(
                label,
                "sampling_kwargs.save_output",
                "output_saving_must_be_explicitly_disabled",
                actual=sampling_kwargs.get("save_output"),
            )
        )

    runtime_provenance = report.get("runtime_provenance")
    if isinstance(runtime_provenance, dict):
        for provenance_field in ("python", "torch", "cuda"):
            if not runtime_provenance.get(provenance_field):
                failures.append(
                    _failure(
                        label,
                        f"runtime_provenance.{provenance_field}",
                        "missing_or_empty",
                    )
                )

    candidate_generation = report.get("candidate_generation")
    hit_counts = (
        candidate_generation.get("per_run_wan_hybrid_hit_count")
        if isinstance(candidate_generation, dict)
        else None
    )
    if not isinstance(hit_counts, list) or not hit_counts:
        failures.append(
            _failure(
                label,
                "candidate_generation.per_run_wan_hybrid_hit_count",
                "missing_candidate_hit_counts",
            )
        )
    else:
        measure_runs = report.get("measure_runs")
        if isinstance(measure_runs, int) and len(hit_counts) != measure_runs:
            failures.append(
                _failure(
                    label,
                    "candidate_generation.per_run_wan_hybrid_hit_count",
                    "candidate_hit_count_length_mismatch",
                    expected=measure_runs,
                    actual=len(hit_counts),
                )
            )
        for run_index, hit_count in enumerate(hit_counts):
            if (
                isinstance(hit_count, bool)
                or not isinstance(hit_count, int)
                or hit_count <= 0
            ):
                failures.append(
                    _failure(
                        label,
                        (
                            "candidate_generation.per_run_wan_hybrid_hit_count"
                            f"[{run_index}]"
                        ),
                        "candidate_hit_count_not_positive",
                        actual=hit_count,
                    )
                )

    performance = report.get("performance")
    speedup = (
        performance.get("wall_median_speedup")
        if isinstance(performance, dict)
        else None
    )
    speedup_min = MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]
    if (
        isinstance(speedup, bool)
        or not isinstance(speedup, (int, float))
        or not math.isfinite(speedup)
    ):
        failures.append(
            _failure(
                label,
                "performance.wall_median_speedup",
                "missing_or_non_finite",
                actual=speedup,
            )
        )
    elif speedup < speedup_min:
        failures.append(
            _failure(
                label,
                "performance.wall_median_speedup",
                "speedup_below_minimum",
                minimum=speedup_min,
                actual=speedup,
            )
        )
    return failures


def aggregate_paired_performance_reports(
    reference_first: Any,
    candidate_first: Any,
) -> dict[str, Any]:
    """Validate and aggregate one reference-first and one candidate-first run."""
    failures = _validate_report(
        reference_first,
        label="reference_first",
        expected_order="reference-first",
    )
    failures.extend(
        _validate_report(
            candidate_first,
            label="candidate_first",
            expected_order="candidate-first",
        )
    )

    if isinstance(reference_first, dict) and isinstance(candidate_first, dict):
        for field in (*_CONFIGURATION_FIELDS, *_IDENTITY_FIELDS):
            lhs = reference_first.get(field)
            rhs = candidate_first.get(field)
            if lhs is None or rhs is None:
                continue
            if lhs != rhs:
                failures.append(
                    {
                        "report": "paired",
                        "field": field,
                        "reason": "paired_value_mismatch",
                        "reference_first_sha256": _canonical_sha256(lhs),
                        "candidate_first_sha256": _canonical_sha256(rhs),
                    }
                )

    def _speedup(report: Any) -> float | None:
        if not isinstance(report, dict):
            return None
        performance = report.get("performance")
        if not isinstance(performance, dict):
            return None
        value = performance.get("wall_median_speedup")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    return {
        "schema_version": 1,
        "passed": not failures,
        "thresholds": {
            "wall_median_speedup_min": MODEL_QUALIFICATION_THRESHOLDS[
                "speedup_min"
            ],
            "candidate_hit_count_min_exclusive": 0,
            "required_orders": ["reference-first", "candidate-first"],
            "trajectory_capture_enabled": False,
        },
        "paired_speedups": {
            "reference_first": _speedup(reference_first),
            "candidate_first": _speedup(candidate_first),
        },
        "failures": failures,
    }


def _load_report(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(report, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return report, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-first-json", required=True, type=Path)
    parser.add_argument("--candidate-first-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reference_first, reference_first_sha256 = _load_report(
            args.reference_first_json.expanduser().resolve()
        )
        candidate_first, candidate_first_sha256 = _load_report(
            args.candidate_first_json.expanduser().resolve()
        )
        result = aggregate_paired_performance_reports(
            reference_first, candidate_first
        )
        result["input_artifacts"] = {
            "reference_first": {
                "path": str(args.reference_first_json.expanduser().resolve()),
                "sha256": reference_first_sha256,
            },
            "candidate_first": {
                "path": str(args.candidate_first_json.expanduser().resolve()),
                "sha256": candidate_first_sha256,
            },
        }
    except (OSError, ValueError) as error:
        result = {
            "schema_version": 1,
            "passed": False,
            "thresholds": {},
            "paired_speedups": {},
            "failures": [
                {
                    "report": "input",
                    "field": "$",
                    "reason": "input_load_failed",
                    "detail": str(error),
                }
            ],
        }

    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
