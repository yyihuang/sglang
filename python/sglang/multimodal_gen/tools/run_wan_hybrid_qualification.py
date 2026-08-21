"""Run and gate the public Wan hybrid model qualification protocol.

The runner is intentionally separate from cluster or container staging.  A
caller provides the staged SGLang and FlashInfer revisions as evidence, while
this module owns the model-level execution matrix and report validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MIN_QUALIFICATION_MEASURE_RUNS,
    MIN_QUALIFICATION_WARMUP_RUNS,
    MODEL_QUALIFICATION_THRESHOLDS,
    QUALIFICATION_RUN_ORDERS,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    validate_wan_transformer_forward_report,
)


QUALIFICATION_SCENARIOS = ("single-block", "full-transformer", "generation")
QUALIFICATION_MODES = ("correctness", "performance")
SCENARIO_EVIDENCE_SCOPES = {
    "single-block": "generation-trajectory-selected-transformer-block",
    "full-transformer": "generation-trajectory-primary-transformer-component",
    "generation": "generation-trajectory-all-eligible-transformer-components",
}
FULL_TRANSFORMER_FORWARD_EVIDENCE_SCOPE = (
    "independent-single-forward-all-transformer-blocks"
)


@dataclass(frozen=True)
class WanQualificationConfig:
    model_path: str
    model_id: str | None
    output_dir: Path
    sglang_revision: str
    flashinfer_revision: str
    staging_label: str
    scenarios: tuple[str, ...] = QUALIFICATION_SCENARIOS
    modes: tuple[str, ...] = QUALIFICATION_MODES
    prompt: str = "A curious raccoon"
    width: int = 640
    height: int = 384
    num_frames: int = 17
    num_inference_steps: int = 12
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    seed: int = 0
    num_gpus: int = 1
    warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS
    measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS
    master_port: int = 30000
    reference_fp4_gemm_backend: str = "flashinfer_trtllm"
    candidate_fp4_gemm_backend: str = "flashinfer_trtllm"
    python_executable: str = sys.executable
    extra_compare_args: tuple[str, ...] = ()
    full_transformer_forward_reports: tuple[Path, ...] = ()

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must be non-empty")
        for name, value in (
            ("sglang_revision", self.sglang_revision),
            ("flashinfer_revision", self.flashinfer_revision),
            ("staging_label", self.staging_label),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        invalid_scenarios = set(self.scenarios) - set(QUALIFICATION_SCENARIOS)
        if invalid_scenarios:
            raise ValueError(f"unsupported scenarios: {sorted(invalid_scenarios)}")
        invalid_modes = set(self.modes) - set(QUALIFICATION_MODES)
        if invalid_modes:
            raise ValueError(f"unsupported modes: {sorted(invalid_modes)}")
        if self.warmup_runs < MIN_QUALIFICATION_WARMUP_RUNS:
            raise ValueError(
                f"warmup_runs must be >= {MIN_QUALIFICATION_WARMUP_RUNS}"
            )
        if self.measure_runs < MIN_QUALIFICATION_MEASURE_RUNS:
            raise ValueError(
                f"measure_runs must be >= {MIN_QUALIFICATION_MEASURE_RUNS}"
            )
        if self.master_port <= 0:
            raise ValueError("master_port must be positive")


@dataclass(frozen=True)
class QualificationInvocation:
    scenario: str
    evidence_scope: str
    comparison_mode: str
    run_order: str
    output_path: Path
    command: tuple[str, ...]


def _scenario_candidate_args(scenario: str) -> tuple[str, ...]:
    if scenario == "single-block":
        return (
            "--candidate-attention-backend",
            "wan_hybrid",
            "--candidate-attention-backend-config",
            '{"wan_hybrid_layer_indices":[0]}',
        )
    if scenario == "full-transformer":
        return (
            "--candidate-attention-backend",
            "fa",
            "--candidate-component-attention-backend",
            "transformer=wan_hybrid",
            "--candidate-component-attention-backend",
            "transformer_2=fa",
        )
    if scenario == "generation":
        return ("--candidate-attention-backend", "wan_hybrid")
    raise ValueError(f"unsupported scenario: {scenario}")


def build_qualification_plan(
    config: WanQualificationConfig,
) -> list[QualificationInvocation]:
    config.validate()
    common = [
        config.python_executable,
        "-m",
        "sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity",
        "--model-path",
        config.model_path,
        "--prompt",
        config.prompt,
        "--width",
        str(config.width),
        "--height",
        str(config.height),
        "--num-frames",
        str(config.num_frames),
        "--num-inference-steps",
        str(config.num_inference_steps),
        "--guidance-scale",
        str(config.guidance_scale),
        "--guidance-scale-2",
        str(config.guidance_scale_2),
        "--seed",
        str(config.seed),
        "--num-gpus",
        str(config.num_gpus),
        "--warmup-runs",
        str(config.warmup_runs),
        "--measure-runs",
        str(config.measure_runs),
        "--reference-attention-backend",
        "fa",
        "--reference-fp4-gemm-backend",
        config.reference_fp4_gemm_backend,
        "--candidate-fp4-gemm-backend",
        config.candidate_fp4_gemm_backend,
        "--enforce-qualification",
    ]
    if config.model_id is not None:
        common.extend(("--model-id", config.model_id))
    common.extend(config.extra_compare_args)

    invocations: list[QualificationInvocation] = []
    next_port = config.master_port
    for scenario in config.scenarios:
        candidate_args = _scenario_candidate_args(scenario)
        for comparison_mode in config.modes:
            run_orders = (
                QUALIFICATION_RUN_ORDERS
                if comparison_mode == "correctness"
                else ("both",)
            )
            for run_order in run_orders:
                suffix = (
                    f"{comparison_mode}-{run_order}"
                    if comparison_mode == "correctness"
                    else "performance-both-orders"
                )
                output_path = config.output_dir / f"wan-{scenario}-{suffix}.json"
                command = tuple(
                    common
                    + list(candidate_args)
                    + [
                        "--comparison-mode",
                        comparison_mode,
                        "--run-order",
                        run_order,
                        "--master-port",
                        str(next_port),
                        "--output-json",
                        str(output_path),
                    ]
                )
                invocations.append(
                    QualificationInvocation(
                        scenario=scenario,
                        evidence_scope=SCENARIO_EVIDENCE_SCOPES[scenario],
                        comparison_mode=comparison_mode,
                        run_order=run_order,
                        output_path=output_path,
                        command=command,
                    )
                )
                next_port += 1
    return invocations


def validate_full_transformer_forward_evidence(
    reports: Sequence[Any],
    *,
    expected_warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    expected_measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
) -> list[str]:
    """Require one valid independent forward report for each execution order."""

    errors = []
    if len(reports) != len(QUALIFICATION_RUN_ORDERS):
        errors.append(
            "independent full-transformer evidence requires exactly two reports"
        )
    run_orders = []
    for report_index, report in enumerate(reports):
        location = f"full_transformer_forward_reports[{report_index}]"
        report_errors = validate_wan_transformer_forward_report(
            report,
            expected_warmup_runs=expected_warmup_runs,
            expected_measure_runs=expected_measure_runs,
        )
        errors.extend(f"{location}: {error}" for error in report_errors)
        if isinstance(report, dict):
            run_orders.append(report.get("run_order"))
    if set(run_orders) != set(QUALIFICATION_RUN_ORDERS) or len(run_orders) != len(
        QUALIFICATION_RUN_ORDERS
    ):
        errors.append(
            "independent full-transformer evidence must cover both execution orders"
        )
    return errors


def _full_transformer_forward_evidence(
    config: WanQualificationConfig, *, read_reports: bool
) -> dict[str, Any]:
    required = "full-transformer" in config.scenarios
    evidence: dict[str, Any] = {
        "required": required,
        "scope": FULL_TRANSFORMER_FORWARD_EVIDENCE_SCOPE,
        "expected_run_orders": list(QUALIFICATION_RUN_ORDERS),
        "report_paths": [
            str(path) for path in config.full_transformer_forward_reports
        ],
    }
    if not required:
        evidence["validation_status"] = "not-required"
        evidence["validation_errors"] = []
        return evidence
    if not read_reports:
        evidence["validation_status"] = "deferred"
        evidence["validation_errors"] = []
        return evidence

    reports = []
    errors = []
    for report_index, path in enumerate(config.full_transformer_forward_reports):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"full_transformer_forward_reports[{report_index}] is unreadable: "
                f"{exc}"
            )
    evidence["observed_run_orders"] = [
        report.get("run_order") if isinstance(report, dict) else None
        for report in reports
    ]
    if not errors:
        errors.extend(
            validate_full_transformer_forward_evidence(
                reports,
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
            )
        )
    elif len(config.full_transformer_forward_reports) != len(
        QUALIFICATION_RUN_ORDERS
    ):
        errors.append(
            "independent full-transformer evidence requires exactly two reports"
        )
    evidence["validation_status"] = "passed" if not errors else "failed"
    evidence["validation_errors"] = errors
    return evidence


def _validate_generation_summary(
    generation: Any,
    *,
    expected_warmup_runs: int,
    expected_measure_runs: int,
    require_hits: bool,
    location: str,
) -> list[str]:
    if not isinstance(generation, dict):
        return [f"{location}: missing generation summary"]
    errors = []
    if generation.get("warmup_runs") != expected_warmup_runs:
        errors.append(f"{location}: unexpected warmup run count")
    if generation.get("measure_runs") != expected_measure_runs:
        errors.append(f"{location}: unexpected measured run count")
    durations = generation.get("per_run_generation_time_s")
    if not isinstance(durations, list) or len(durations) != expected_measure_runs:
        errors.append(f"{location}: missing measured generation durations")
    if require_hits:
        hit_counts = generation.get("per_run_wan_hybrid_hit_count")
        if (
            not isinstance(hit_counts, list)
            or len(hit_counts) != expected_measure_runs
            or any(
                isinstance(hit_count, bool)
                or not isinstance(hit_count, int)
                or hit_count <= 0
                for hit_count in hit_counts
            )
        ):
            errors.append(
                f"{location}: every measured wan_hybrid hit count must be positive"
            )
    return errors


def _validate_all_step_comparisons(
    comparisons: Any, *, expected_pairs: set[tuple[int, int]], location: str
) -> list[str]:
    if not isinstance(comparisons, list):
        return [f"{location}: missing comparisons"]
    actual_pairs = {
        (comparison.get("reference_run_index"), comparison.get("candidate_run_index"))
        for comparison in comparisons
        if isinstance(comparison, dict)
    }
    errors = []
    if actual_pairs != expected_pairs or len(comparisons) != len(expected_pairs):
        errors.append(f"{location}: run-pair coverage is incomplete")
    for pair_index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            errors.append(f"{location}[{pair_index}]: invalid comparison")
            continue
        trajectory = comparison.get("trajectory_metrics")
        if not isinstance(trajectory, dict):
            errors.append(f"{location}[{pair_index}]: missing trajectory metrics")
            continue
        per_step = trajectory.get("per_step_metrics")
        if (
            not isinstance(per_step, list)
            or not per_step
            or len(per_step) != trajectory.get("num_steps")
        ):
            errors.append(f"{location}[{pair_index}]: all-step coverage is incomplete")
    return errors


def _validate_candidate_hit_qualification(
    qualification: dict[str, Any], *, location: str
) -> list[str]:
    hit_qualification = qualification.get("candidate_backend_hits")
    if (
        not isinstance(hit_qualification, dict)
        or hit_qualification.get("passed") is not True
        or hit_qualification.get("failures") != []
        or hit_qualification.get("thresholds")
        != {"candidate_hit_count_min_exclusive": 0}
    ):
        return [f"{location}: candidate backend-hit qualification is incomplete"]
    return []


def validate_qualification_report(
    report: Any,
    invocation: QualificationInvocation,
    config: WanQualificationConfig,
) -> list[str]:
    if not isinstance(report, dict):
        return ["report is not a JSON object"]
    errors = []
    if report.get("comparison_mode") != invocation.comparison_mode:
        errors.append("comparison_mode does not match the invocation")
    if report.get("run_order") != invocation.run_order:
        errors.append("run_order does not match the invocation")
    if report.get("warmup_runs") != config.warmup_runs:
        errors.append("report warmup_runs does not match the protocol")
    if report.get("measure_runs") != config.measure_runs:
        errors.append("report measure_runs does not match the protocol")
    qualification = report.get("qualification")
    if not isinstance(qualification, dict) or qualification.get("passed") is not True:
        errors.append("qualification did not pass")
    elif qualification.get("failures") != []:
        errors.append("qualification failures are not empty")

    server_kwargs = report.get("server_kwargs")
    if not isinstance(server_kwargs, dict):
        errors.append("missing server_kwargs")
    else:
        reference_server = server_kwargs.get("reference")
        candidate_server = server_kwargs.get("candidate")
        if (
            not isinstance(reference_server, dict)
            or reference_server.get("attention_backend") != "fa"
        ):
            errors.append("reference is not the production FA backend")
        if not isinstance(candidate_server, dict):
            errors.append("missing candidate server configuration")
        elif invocation.scenario in ("single-block", "generation"):
            if candidate_server.get("attention_backend") != "wan_hybrid":
                errors.append("candidate is not the public wan_hybrid backend")
        else:
            if candidate_server.get("attention_backend") != "fa":
                errors.append("full-transformer candidate global backend is not FA")
            if candidate_server.get("component_attention_backends") != {
                "transformer": "wan_hybrid",
                "transformer_2": "fa",
            }:
                errors.append("full-transformer component routing is incomplete")

    sampling_kwargs = report.get("sampling_kwargs")
    if not isinstance(sampling_kwargs, dict):
        errors.append("missing sampling_kwargs")
    elif invocation.comparison_mode == "correctness":
        if sampling_kwargs.get("return_trajectory_latents") is not True:
            errors.append("correctness did not capture trajectory latents")
    elif sampling_kwargs.get("return_trajectory_latents") is not False:
        errors.append("performance unexpectedly captured trajectory latents")

    if invocation.comparison_mode == "correctness":
        if isinstance(qualification, dict):
            if qualification.get("thresholds") != MODEL_QUALIFICATION_THRESHOLDS:
                errors.append("correctness thresholds do not match the protocol")
            errors.extend(
                _validate_candidate_hit_qualification(
                    qualification, location="qualification"
                )
            )
        errors.extend(
            _validate_generation_summary(
                report.get("reference_generation"),
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
                require_hits=False,
                location="reference_generation",
            )
        )
        errors.extend(
            _validate_generation_summary(
                report.get("candidate_generation"),
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
                require_hits=True,
                location="candidate_generation",
            )
        )
        cross = report.get("cross_variant_metrics")
        expected_cross_pairs = set(
            itertools.product(
                range(config.measure_runs), range(config.measure_runs)
            )
        )
        if not isinstance(cross, dict) or cross.get("pairing") != "cross-product":
            errors.append("cross-variant comparison is not a cross-product")
        else:
            errors.extend(
                _validate_all_step_comparisons(
                    cross.get("comparisons"),
                    expected_pairs=expected_cross_pairs,
                    location="cross_variant_metrics.comparisons",
                )
            )
        repeatability = report.get("repeatability")
        expected_repeat_pairs = set(
            itertools.combinations(range(config.measure_runs), 2)
        )
        if not isinstance(repeatability, dict):
            errors.append("missing repeatability")
        else:
            for variant in ("reference", "candidate"):
                summary = repeatability.get(variant)
                location = f"repeatability.{variant}"
                if (
                    not isinstance(summary, dict)
                    or summary.get("available") is not True
                    or summary.get("pairing") != "all-pairs"
                ):
                    errors.append(f"{location}: incomplete all-pairs summary")
                    continue
                errors.extend(
                    _validate_all_step_comparisons(
                        summary.get("comparisons"),
                        expected_pairs=expected_repeat_pairs,
                        location=f"{location}.comparisons",
                    )
                )
    else:
        if "cross_variant_metrics" in report or "repeatability" in report:
            errors.append("performance report contains trajectory comparisons")
        order_results = report.get("order_results")
        if not isinstance(order_results, dict):
            errors.append("missing dual-order performance results")
        else:
            for run_order in QUALIFICATION_RUN_ORDERS:
                order_result = order_results.get(run_order)
                if not isinstance(order_result, dict):
                    errors.append(f"missing performance result for {run_order}")
                    continue
                errors.extend(
                    _validate_generation_summary(
                        order_result.get("reference_generation"),
                        expected_warmup_runs=config.warmup_runs,
                        expected_measure_runs=config.measure_runs,
                        require_hits=False,
                        location=f"order_results.{run_order}.reference_generation",
                    )
                )
                errors.extend(
                    _validate_generation_summary(
                        order_result.get("candidate_generation"),
                        expected_warmup_runs=config.warmup_runs,
                        expected_measure_runs=config.measure_runs,
                        require_hits=True,
                        location=f"order_results.{run_order}.candidate_generation",
                    )
                )
                speedup = order_result.get("performance", {}).get(
                    "wall_median_speedup"
                )
                if (
                    isinstance(speedup, bool)
                    or not isinstance(speedup, (int, float))
                    or not math.isfinite(speedup)
                    or speedup < MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]
                ):
                    errors.append(f"{run_order}: wall-median speedup is below 1.0")
        if isinstance(qualification, dict):
            expected_thresholds = {
                "required_run_orders": list(QUALIFICATION_RUN_ORDERS),
                "warmup_runs_min": MIN_QUALIFICATION_WARMUP_RUNS,
                "measure_runs_min": MIN_QUALIFICATION_MEASURE_RUNS,
                "speedup_min": MODEL_QUALIFICATION_THRESHOLDS["speedup_min"],
                "candidate_hit_count_min_exclusive": 0,
            }
            if qualification.get("thresholds") != expected_thresholds:
                errors.append("performance thresholds do not match the protocol")
    return errors


def run_qualification_plan(
    config: WanQualificationConfig, *, dry_run: bool = False
) -> dict[str, Any]:
    plan = build_qualification_plan(config)
    forward_evidence = _full_transformer_forward_evidence(
        config, read_reports=not dry_run
    )
    if dry_run:
        return {
            "schema_version": 2,
            "staging": {
                "label": config.staging_label,
                "sglang_revision": config.sglang_revision,
                "flashinfer_revision": config.flashinfer_revision,
            },
            "full_transformer_forward_evidence": forward_evidence,
            "invocations": [
                {
                    "scenario": invocation.scenario,
                    "evidence_scope": invocation.evidence_scope,
                    "comparison_mode": invocation.comparison_mode,
                    "run_order": invocation.run_order,
                    "output_path": str(invocation.output_path),
                    "command": list(invocation.command),
                }
                for invocation in plan
            ],
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "wan-hybrid-qualification-manifest.json"
    if forward_evidence["validation_status"] == "failed":
        manifest = {
            "schema_version": 2,
            "passed": False,
            "staging": {
                "label": config.staging_label,
                "sglang_revision": config.sglang_revision,
                "flashinfer_revision": config.flashinfer_revision,
            },
            "model_path": config.model_path,
            "model_id": config.model_id,
            "warmup_runs": config.warmup_runs,
            "measure_runs": config.measure_runs,
            "full_transformer_forward_evidence": forward_evidence,
            "invocations": [
                {
                    "scenario": invocation.scenario,
                    "evidence_scope": invocation.evidence_scope,
                    "comparison_mode": invocation.comparison_mode,
                    "run_order": invocation.run_order,
                    "output_path": str(invocation.output_path),
                    "command": list(invocation.command),
                    "status": "not-run-invalid-forward-evidence",
                }
                for invocation in plan
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(
            "Independent full-transformer evidence failed validation; "
            f"see {manifest_path}"
        )

    records = []
    passed = True
    for invocation in plan:
        completed = subprocess.run(invocation.command, check=False)
        errors = []
        if completed.returncode != 0:
            errors.append(f"comparison process exited with {completed.returncode}")
        if not invocation.output_path.is_file():
            errors.append("comparison report was not written")
        else:
            try:
                report = json.loads(invocation.output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"comparison report is unreadable: {exc}")
            else:
                errors.extend(validate_qualification_report(report, invocation, config))
        passed = passed and not errors
        records.append(
            {
                "scenario": invocation.scenario,
                "evidence_scope": invocation.evidence_scope,
                "comparison_mode": invocation.comparison_mode,
                "run_order": invocation.run_order,
                "output_path": str(invocation.output_path),
                "command": list(invocation.command),
                "returncode": completed.returncode,
                "validation_errors": errors,
            }
        )

    manifest = {
        "schema_version": 2,
        "passed": passed,
        "staging": {
            "label": config.staging_label,
            "sglang_revision": config.sglang_revision,
            "flashinfer_revision": config.flashinfer_revision,
        },
        "model_path": config.model_path,
        "model_id": config.model_id,
        "warmup_runs": config.warmup_runs,
        "measure_runs": config.measure_runs,
        "full_transformer_forward_evidence": forward_evidence,
        "invocations": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Wan hybrid qualification failed; see {manifest_path}")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sglang-revision", required=True)
    parser.add_argument("--flashinfer-revision", required=True)
    parser.add_argument("--staging-label", required=True)
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=QUALIFICATION_SCENARIOS,
        required=True,
        help="Scenario to run; repeat to select more than one.",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=QUALIFICATION_MODES,
        required=True,
        help="Qualification mode; repeat to select both.",
    )
    parser.add_argument("--prompt", default="A curious raccoon")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--warmup-runs", type=int, default=MIN_QUALIFICATION_WARMUP_RUNS
    )
    parser.add_argument(
        "--measure-runs", type=int, default=MIN_QUALIFICATION_MEASURE_RUNS
    )
    parser.add_argument("--master-port", type=int, default=30000)
    parser.add_argument(
        "--reference-fp4-gemm-backend", default="flashinfer_trtllm"
    )
    parser.add_argument(
        "--candidate-fp4-gemm-backend", default="flashinfer_trtllm"
    )
    parser.add_argument(
        "--compare-arg",
        dest="extra_compare_args",
        action="append",
        default=[],
        help="Additional single argument forwarded to each comparison invocation.",
    )
    parser.add_argument(
        "--full-transformer-forward-report",
        dest="full_transformer_forward_reports",
        action="append",
        type=Path,
        default=[],
        help=(
            "Independent single-forward correctness report; pass one for each "
            "execution order when selecting the full-transformer scenario."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = WanQualificationConfig(
        model_path=args.model_path,
        model_id=args.model_id,
        output_dir=args.output_dir.expanduser().resolve(),
        sglang_revision=args.sglang_revision,
        flashinfer_revision=args.flashinfer_revision,
        staging_label=args.staging_label,
        scenarios=tuple(args.scenarios or QUALIFICATION_SCENARIOS),
        modes=tuple(args.modes or QUALIFICATION_MODES),
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        seed=args.seed,
        num_gpus=args.num_gpus,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        master_port=args.master_port,
        reference_fp4_gemm_backend=args.reference_fp4_gemm_backend,
        candidate_fp4_gemm_backend=args.candidate_fp4_gemm_backend,
        extra_compare_args=tuple(args.extra_compare_args),
        full_transformer_forward_reports=tuple(
            path.expanduser().resolve()
            for path in args.full_transformer_forward_reports
        ),
    )
    manifest = run_qualification_plan(config, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
