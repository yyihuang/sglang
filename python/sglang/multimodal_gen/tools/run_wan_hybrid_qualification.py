"""Run and gate the public Wan hybrid model qualification protocol.

The runner is intentionally separate from cluster or container staging.  A
caller provides the staged SGLang and FlashInfer revisions as evidence, while
this module owns the model-level execution matrix and report validation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    validate_wan_hybrid_exact_serving_boundary_evidence,
)
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MIN_QUALIFICATION_MEASURE_RUNS,
    MIN_QUALIFICATION_WARMUP_RUNS,
    MODEL_QUALIFICATION_THRESHOLDS,
    QUALIFICATION_RUN_ORDERS,
    WAN_HYBRID_PROMOTION_GENERATION_HITS,
    WAN_HYBRID_PROMOTION_LAYER_INDICES,
    WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
    validate_reference_attention_backend_identity,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    validate_wan_transformer_forward_performance_report,
    validate_wan_transformer_forward_report,
)

QUALIFICATION_SCENARIOS = ("single-block", "full-transformer", "generation")
QUALIFICATION_MODES = ("correctness", "performance")
WAN_TRANSFORMER_COMPONENTS = ("transformer", "transformer_2")
WAN_HYBRID_DEFAULT_LAYER_INDICES = [39]
WAN_HYBRID_SINGLE_BLOCK_LAYER_INDICES = [WAN_HYBRID_PROMOTION_LAYER_INDICES[0]]
FORBIDDEN_VARIANT_MODEL_OVERRIDE_OPTIONS = frozenset(
    {
        "--reference-transformer-path",
        "--candidate-transformer-path",
        "--reference-component-path",
        "--candidate-component-path",
    }
)
QUALIFICATION_OWNED_PORT_OPTIONS = frozenset(
    {
        "--master-port",
        "--reference-scheduler-port",
        "--candidate-scheduler-port",
        "--http-port",
    }
)
SCENARIO_EVIDENCE_SCOPES = {
    "single-block": "generation-trajectory-selected-transformer-block",
    "full-transformer": "independent-direct-correctness-and-trajectory-off-performance",
    "generation": "generation-trajectory-all-eligible-transformer-components",
}
FULL_TRANSFORMER_FORWARD_EVIDENCE_SCOPE = (
    "independent-direct-correctness-and-trajectory-off-performance"
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
    full_transformer_performance_reports: tuple[Path, ...] = ()

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
        if "performance" in self.modes and (
            self.warmup_runs != MIN_QUALIFICATION_WARMUP_RUNS
            or self.measure_runs != MIN_QUALIFICATION_MEASURE_RUNS
        ):
            raise ValueError(
                "performance qualification requires exactly warmup=2/measure=5"
            )
        if self.warmup_runs < MIN_QUALIFICATION_WARMUP_RUNS:
            raise ValueError(f"warmup_runs must be >= {MIN_QUALIFICATION_WARMUP_RUNS}")
        if self.measure_runs < MIN_QUALIFICATION_MEASURE_RUNS:
            raise ValueError(
                f"measure_runs must be >= {MIN_QUALIFICATION_MEASURE_RUNS}"
            )
        if self.master_port <= 0:
            raise ValueError("master_port must be positive")
        override_options = {
            argument.split("=", 1)[0]
            for argument in self.extra_compare_args
            if argument.startswith("--")
        }
        forbidden = sorted(override_options & FORBIDDEN_VARIANT_MODEL_OVERRIDE_OPTIONS)
        if forbidden:
            raise ValueError(
                "qualification forbids reference/candidate model overrides: "
                + ", ".join(forbidden)
            )
        port_overrides = sorted(override_options & QUALIFICATION_OWNED_PORT_OPTIONS)
        if port_overrides:
            raise ValueError(
                "qualification owns invocation port options: "
                + ", ".join(port_overrides)
            )


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
            json.dumps(
                {
                    "wan_hybrid_layer_indices": WAN_HYBRID_SINGLE_BLOCK_LAYER_INDICES,
                    "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if scenario == "full-transformer":
        raise ValueError(
            "full-transformer promotion uses independent direct evidence, not a "
            "generation-trajectory routing surrogate"
        )
    if scenario == "generation":
        return (
            "--candidate-attention-backend",
            "wan_hybrid",
            "--candidate-attention-backend-config",
            json.dumps(
                {
                    "wan_hybrid_layer_indices": WAN_HYBRID_PROMOTION_LAYER_INDICES,
                    "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
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
        if scenario == "full-transformer":
            continue
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
                        "--reference-scheduler-port",
                        str(next_port + 1),
                        "--candidate-scheduler-port",
                        str(next_port + 2),
                        "--http-port",
                        str(next_port + 3),
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
                next_port += 4
    if next_port - 1 > 65535:
        raise ValueError("qualification port range exceeds 65535")
    return invocations


def validate_full_transformer_forward_evidence(
    reports: Sequence[Any],
    *,
    expected_warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    expected_measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
    expected_model_path: str | Path | None = None,
) -> list[str]:
    """Require valid forwards for both 40-block components in both orders."""

    errors = []
    expected_evidence = set(
        itertools.product(WAN_TRANSFORMER_COMPONENTS, QUALIFICATION_RUN_ORDERS)
    )
    if len(reports) != len(expected_evidence):
        errors.append(
            "independent full-transformer evidence requires exactly four reports"
        )
    observed_evidence = []
    for report_index, report in enumerate(reports):
        location = f"full_transformer_forward_reports[{report_index}]"
        report_errors = validate_wan_transformer_forward_report(
            report,
            expected_warmup_runs=expected_warmup_runs,
            expected_measure_runs=expected_measure_runs,
            expected_model_path=expected_model_path,
        )
        errors.extend(f"{location}: {error}" for error in report_errors)
        if isinstance(report, dict):
            observed_evidence.append(
                (report.get("component_name"), report.get("run_order"))
            )
    if set(observed_evidence) != expected_evidence or len(observed_evidence) != len(
        expected_evidence
    ):
        errors.append(
            "independent full-transformer evidence must cover transformer and "
            "transformer_2 in both execution orders"
        )
    bindings_by_component: dict[str, list[Any]] = {
        component: [] for component in WAN_TRANSFORMER_COMPONENTS
    }
    for report in reports:
        if (
            isinstance(report, dict)
            and report.get("component_name") in bindings_by_component
        ):
            bindings_by_component[report["component_name"]].append(
                report.get("evidence_binding")
            )
    for component, bindings in bindings_by_component.items():
        if len(bindings) == 2 and bindings[0] != bindings[1]:
            errors.append(
                f"{component} evidence binding differs between execution orders"
            )
    resolved_paths = {
        binding.get("resolved_component_path")
        for bindings in bindings_by_component.values()
        for binding in bindings[:1]
        if isinstance(binding, dict)
    }
    if len(resolved_paths) != len(WAN_TRANSFORMER_COMPONENTS):
        errors.append("transformer component evidence paths are not distinct")
    return errors


def _full_transformer_forward_evidence(
    config: WanQualificationConfig, *, read_reports: bool
) -> dict[str, Any]:
    required = "full-transformer" in config.scenarios
    evidence: dict[str, Any] = {
        "required": required,
        "scope": FULL_TRANSFORMER_FORWARD_EVIDENCE_SCOPE,
        "expected_run_orders": list(QUALIFICATION_RUN_ORDERS),
        "expected_components": list(WAN_TRANSFORMER_COMPONENTS),
        "report_paths": [str(path) for path in config.full_transformer_forward_reports],
        "performance_report_paths": [
            str(path) for path in config.full_transformer_performance_reports
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
    evidence["observed_component_run_orders"] = [
        (
            {
                "component_name": report.get("component_name"),
                "run_order": report.get("run_order"),
            }
            if isinstance(report, dict)
            else None
        )
        for report in reports
    ]
    if not errors:
        errors.extend(
            validate_full_transformer_forward_evidence(
                reports,
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
                expected_model_path=config.model_path,
            )
        )
    elif len(config.full_transformer_forward_reports) != len(
        WAN_TRANSFORMER_COMPONENTS
    ) * len(QUALIFICATION_RUN_ORDERS):
        errors.append(
            "independent full-transformer evidence requires exactly four reports"
        )
    performance_reports = []
    if "performance" in config.modes:
        if len(config.full_transformer_performance_reports) != 1:
            errors.append(
                "independent full-transformer performance requires exactly one "
                "trajectory-off dual-order report"
            )
        else:
            path = config.full_transformer_performance_reports[0]
            try:
                performance_report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    "full_transformer_performance_reports[0] is unreadable: " f"{exc}"
                )
            else:
                performance_reports.append(performance_report)
                errors.extend(
                    f"full_transformer_performance_reports[0]: {error}"
                    for error in validate_wan_transformer_forward_performance_report(
                        performance_report
                    )
                )
                binding = (
                    performance_report.get("evidence_binding")
                    if isinstance(performance_report, dict)
                    else None
                )
                expected_component_path = str(
                    (Path(config.model_path).expanduser().resolve() / "transformer_2")
                )
                if (
                    not isinstance(binding, dict)
                    or binding.get("resolved_component_path") != expected_component_path
                ):
                    errors.append(
                        "full_transformer_performance_reports[0]: resolved component "
                        "path does not match model/transformer_2"
                    )
    evidence["observed_performance_reports"] = len(performance_reports)
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
    scenario: str,
    expected_num_steps: int,
    expected_num_frames: int,
    expected_frame_shape: tuple[int, int, int],
) -> list[str]:
    if not isinstance(generation, dict):
        return [f"{location}: missing generation summary"]
    errors = []
    if generation.get("warmup_runs") != expected_warmup_runs:
        errors.append(f"{location}: unexpected warmup run count")
    if generation.get("measure_runs") != expected_measure_runs:
        errors.append(f"{location}: unexpected measured run count")
    for field_name in (
        "per_run_generation_time_s",
        "per_run_total_duration_ms",
    ):
        durations = generation.get(field_name)
        if (
            not isinstance(durations, list)
            or len(durations) != expected_measure_runs
            or any(
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
                for duration in durations
            )
        ):
            errors.append(
                f"{location}: {field_name} must contain one finite positive "
                "duration per measured run"
            )
    if generation.get("timer_scope") != (
        "complete DiffGenerator.generate call including output materialization"
    ):
        errors.append(f"{location}: timer_scope does not include complete generation")
    output_summaries = generation.get("per_run_output_summaries")
    if (
        not isinstance(output_summaries, list)
        or len(output_summaries) != expected_measure_runs
    ):
        errors.append(f"{location}: missing per-run output summaries")
    else:
        for run_index, output in enumerate(output_summaries):
            output_location = f"{location}.per_run_output_summaries[{run_index}]"
            if not isinstance(output, dict):
                errors.append(f"{output_location}: invalid output summary")
                continue
            digest = output.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                errors.append(f"{output_location}: invalid SHA256 digest")
            if output.get("finite") is not True:
                errors.append(f"{output_location}: output is not finite")
            if output.get("num_frames") != expected_num_frames:
                errors.append(f"{output_location}: unexpected frame count")
            shapes = output.get("frame_shapes")
            dtypes = output.get("frame_dtypes")
            if shapes != [list(expected_frame_shape)] * expected_num_frames:
                errors.append(f"{output_location}: unexpected frame shapes")
            if dtypes != ["uint8"] * expected_num_frames:
                errors.append(f"{output_location}: unexpected frame dtypes")
    if require_hits:
        request_ids = generation.get("per_run_request_id")
        hit_counts = generation.get("per_run_wan_hybrid_hit_count")
        expected_hit_counts = generation.get("per_run_wan_hybrid_expected_hit_count")
        coverages = generation.get("per_run_wan_hybrid_coverage")
        if (
            not isinstance(request_ids, list)
            or len(request_ids) != expected_measure_runs
            or any(not isinstance(item, str) or not item for item in request_ids)
            or len(set(request_ids)) != expected_measure_runs
            or not isinstance(hit_counts, list)
            or len(hit_counts) != expected_measure_runs
            or not isinstance(expected_hit_counts, list)
            or len(expected_hit_counts) != expected_measure_runs
            or not isinstance(coverages, list)
            or len(coverages) != expected_measure_runs
        ):
            errors.append(f"{location}: missing exact per-run wan_hybrid hit evidence")
        else:
            for run_index, (
                request_id,
                hit_count,
                expected_hit_count,
                coverage,
            ) in enumerate(
                zip(request_ids, hit_counts, expected_hit_counts, coverages)
            ):
                coverage_location = (
                    f"{location}.per_run_wan_hybrid_coverage[{run_index}]"
                )
                errors.extend(
                    _validate_wan_hybrid_coverage(
                        coverage,
                        scenario=scenario,
                        expected_num_steps=expected_num_steps,
                        location=coverage_location,
                    )
                )
                if (
                    not isinstance(coverage, dict)
                    or coverage.get("request_id") != request_id
                ):
                    errors.append(
                        f"{coverage_location}: request identity does not match result"
                    )
                if not isinstance(coverage, dict):
                    continue
                if (
                    isinstance(hit_count, bool)
                    or not isinstance(hit_count, int)
                    or isinstance(expected_hit_count, bool)
                    or not isinstance(expected_hit_count, int)
                    or expected_hit_count <= 0
                    or hit_count != expected_hit_count
                    or hit_count != coverage.get("actual_hit_count")
                    or expected_hit_count != coverage.get("expected_hit_count")
                ):
                    errors.append(
                        f"{coverage_location}: actual hit count does not equal "
                        "the route-derived expected hit count"
                    )
                if scenario == "generation" and (
                    hit_count != WAN_HYBRID_PROMOTION_GENERATION_HITS
                    or expected_hit_count != WAN_HYBRID_PROMOTION_GENERATION_HITS
                ):
                    errors.append(
                        f"{coverage_location}: generation promotion requires exactly "
                        f"{WAN_HYBRID_PROMOTION_GENERATION_HITS} candidate hits"
                    )
    return errors


def _validate_wan_hybrid_coverage(
    coverage: Any,
    *,
    scenario: str,
    expected_num_steps: int,
    location: str,
) -> list[str]:
    if not isinstance(coverage, dict):
        return [f"{location}: missing coverage object"]
    errors = []
    if coverage.get("schema_version") != 2:
        errors.append(f"{location}: unsupported coverage schema")
    errors.extend(
        f"{location}: {error}"
        for error in validate_wan_hybrid_exact_serving_boundary_evidence(coverage)
    )
    if not isinstance(coverage.get("request_id"), str) or not coverage["request_id"]:
        errors.append(f"{location}: request identity is missing")
    steps = coverage.get("steps")
    if not isinstance(steps, list) or len(steps) != expected_num_steps:
        return errors + [f"{location}: denoising-step coverage is incomplete"]
    if {step.get("step_index") for step in steps if isinstance(step, dict)} != set(
        range(expected_num_steps)
    ):
        errors.append(f"{location}: step indices are incomplete")

    expected_layers = list(range(40))
    expected_hit_count = 0
    actual_hit_count = 0
    route_event_count = 0
    observed_components = set()
    for step_position, step in enumerate(steps):
        step_location = f"{location}.steps[{step_position}]"
        if not isinstance(step, dict):
            errors.append(f"{step_location}: invalid step")
            continue
        component = step.get("active_component")
        if component not in WAN_TRANSFORMER_COMPONENTS:
            errors.append(f"{step_location}: invalid active component")
            continue
        observed_components.add(component)
        actual_timestep = step.get("actual_timestep")
        if isinstance(actual_timestep, bool) or not isinstance(actual_timestep, int):
            errors.append(f"{step_location}: actual timestep is missing")
        branches = step.get("branches")
        if not isinstance(branches, list) or not branches:
            errors.append(f"{step_location}: no executed CFG branches")
            continue
        branch_indices = [
            branch.get("cfg_branch_index")
            for branch in branches
            if isinstance(branch, dict)
        ]
        if (
            branch_indices != list(range(len(branches)))
            or step.get("executed_cfg_branch_indices") != branch_indices
        ):
            errors.append(f"{step_location}: CFG branch evidence is inconsistent")
        for branch_position, branch in enumerate(branches):
            branch_location = f"{step_location}.branches[{branch_position}]"
            if not isinstance(branch, dict):
                errors.append(f"{branch_location}: invalid branch")
                continue
            if (
                branch.get("num_layers") != 40
                or branch.get("layer_indices") != expected_layers
            ):
                errors.append(f"{branch_location}: 40-layer coverage is incomplete")
            if scenario == "generation":
                if (
                    isinstance(actual_timestep, int)
                    and not isinstance(actual_timestep, bool)
                    and actual_timestep <= WAN_HYBRID_PROMOTION_MAX_TIMESTEP
                ):
                    expected_eligible = list(WAN_HYBRID_PROMOTION_LAYER_INDICES)
                else:
                    expected_eligible = []
                expected_fallback = [
                    index for index in expected_layers if index not in expected_eligible
                ]
                expected_control = []
            elif scenario == "full-transformer" and component == "transformer":
                expected_eligible = WAN_HYBRID_DEFAULT_LAYER_INDICES
                expected_fallback = [
                    index for index in expected_layers if index not in expected_eligible
                ]
                expected_control = []
            elif scenario == "single-block":
                if (
                    isinstance(actual_timestep, int)
                    and not isinstance(actual_timestep, bool)
                    and actual_timestep <= WAN_HYBRID_PROMOTION_MAX_TIMESTEP
                ):
                    expected_eligible = list(WAN_HYBRID_SINGLE_BLOCK_LAYER_INDICES)
                else:
                    expected_eligible = []
                expected_fallback = [
                    index for index in expected_layers if index not in expected_eligible
                ]
                expected_control = []
            else:
                expected_eligible = []
                expected_fallback = []
                expected_control = expected_layers
            if (
                branch.get("eligible_layer_indices") != expected_eligible
                or branch.get("planned_hybrid_layer_indices") != expected_eligible
                or branch.get("successful_hybrid_layer_indices") != expected_eligible
                or branch.get("eligible_hybrid_miss_layer_indices") != []
                or branch.get("unexpected_successful_hybrid_layer_indices") != []
                or branch.get("configured_fallback_layer_indices") != expected_fallback
                or branch.get("control_layer_indices") != expected_control
            ):
                errors.append(
                    f"{branch_location}: hybrid/fallback/control routing is invalid"
                )
            branch_expected = len(expected_eligible)
            if (
                branch.get("expected_hit_count") != branch_expected
                or branch.get("actual_hit_count") != branch_expected
            ):
                errors.append(f"{branch_location}: exact hit count mismatch")
            expected_hit_count += branch_expected
            branch_actual = branch.get("actual_hit_count")
            if isinstance(branch_actual, bool) or not isinstance(branch_actual, int):
                branch_actual = 0
            actual_hit_count += branch_actual
            route_event_count += 40

    if observed_components != set(WAN_TRANSFORMER_COMPONENTS):
        errors.append(f"{location}: both Wan transformer components were not exercised")
    if (
        scenario == "generation"
        and expected_hit_count != WAN_HYBRID_PROMOTION_GENERATION_HITS
    ):
        errors.append(
            f"{location}: generation promotion route must derive exactly "
            f"{WAN_HYBRID_PROMOTION_GENERATION_HITS} hits"
        )
    if (
        coverage.get("expected_hit_count") != expected_hit_count
        or coverage.get("actual_hit_count") != actual_hit_count
        or coverage.get("attributed_actual_hit_count") != actual_hit_count
        or coverage.get("unattributed_actual_hit_count") != 0
        or coverage.get("eligible_hybrid_miss_count") != 0
        or coverage.get("num_route_events") != route_event_count
        or coverage.get("num_success_events") != actual_hit_count
    ):
        errors.append(f"{location}: aggregate route/hit evidence is inconsistent")
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


def _validate_report_provenance(
    report: dict[str, Any], config: WanQualificationConfig
) -> list[str]:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        return ["missing qualification provenance"]
    errors = []
    fixed_input = provenance.get("fixed_input")
    digest = provenance.get("input_sha256")
    if not isinstance(fixed_input, dict):
        errors.append("provenance.fixed_input is missing")
    else:
        expected_model_path = str(Path(config.model_path).expanduser().resolve())
        expected_fixed_fields = {
            "model_path": expected_model_path,
            "model_id": config.model_id,
            "prompt": config.prompt,
            "seed": config.seed,
            "sampling_kwargs": report.get("sampling_kwargs"),
        }
        for field_name, expected_value in expected_fixed_fields.items():
            if fixed_input.get(field_name) != expected_value:
                errors.append(f"provenance.fixed_input.{field_name} is inconsistent")
        calculated_digest = hashlib.sha256(
            json.dumps(
                fixed_input, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        if digest != calculated_digest:
            errors.append("provenance input SHA256 is inconsistent")
    model = provenance.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("resolved_path"), str)
        or not model.get("resolved_path")
        or model.get("resolved_path")
        != str(Path(config.model_path).expanduser().resolve())
        or model.get("model_id") != config.model_id
        or not isinstance(model.get("config_files"), list)
        or not model.get("config_files")
    ):
        errors.append("model provenance is incomplete")
    else:
        for config_file in model["config_files"]:
            if (
                not isinstance(config_file, dict)
                or not isinstance(config_file.get("path"), str)
                or not isinstance(config_file.get("sha256"), str)
                or len(config_file["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in config_file["sha256"]
                )
            ):
                errors.append("model config-file provenance is incomplete")
                break
    runtime = provenance.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("sglang_revision") != config.sglang_revision
        or runtime.get("flashinfer_revision") != config.flashinfer_revision
        or not isinstance(runtime.get("gpu"), dict)
        or not runtime.get("gpu", {}).get("name")
    ):
        errors.append("runtime provenance is incomplete or inconsistent")
    else:
        public_api = runtime.get("flashinfer_public_api")
        if not isinstance(public_api, dict) or set(public_api.values()) != {True}:
            errors.append("FlashInfer public wan_hybrid capability is incomplete")
    if provenance.get("normalized_backend_request") != report.get("server_kwargs"):
        errors.append("normalized backend provenance does not match server_kwargs")
    return errors


def _validate_candidate_hit_qualification(
    qualification: dict[str, Any],
    *,
    actual_hit_counts: Any,
    expected_hit_counts: Any,
    location: str,
) -> list[str]:
    hit_qualification = qualification.get("candidate_backend_hits")
    if (
        not isinstance(hit_qualification, dict)
        or hit_qualification.get("passed") is not True
        or hit_qualification.get("failures") != []
        or hit_qualification.get("thresholds")
        != {
            "candidate_hit_count_equals_expected": True,
            "expected_hit_count_min_exclusive": 0,
        }
        or hit_qualification.get("actual_hit_counts") != actual_hit_counts
        or hit_qualification.get("expected_hit_counts") != expected_hit_counts
    ):
        return [f"{location}: candidate backend-hit qualification is incomplete"]
    return []


def _invocation_integer_option(
    invocation: QualificationInvocation, option: str
) -> int | None:
    try:
        index = invocation.command.index(option)
        return int(invocation.command[index + 1])
    except (ValueError, IndexError):
        return None


def _validate_execution_topology(
    report: dict[str, Any], invocation: QualificationInvocation
) -> list[str]:
    topology = report.get("execution_topology")
    if not isinstance(topology, dict):
        return ["missing execution topology"]
    expected_ports = {
        "master_port": _invocation_integer_option(invocation, "--master-port"),
        "reference_scheduler_port": _invocation_integer_option(
            invocation, "--reference-scheduler-port"
        ),
        "candidate_scheduler_port": _invocation_integer_option(
            invocation, "--candidate-scheduler-port"
        ),
        "http_port": _invocation_integer_option(invocation, "--http-port"),
        "reference_strict_ports": True,
        "candidate_strict_ports": True,
    }
    errors = []
    if (
        isinstance(topology.get("controller_pid"), bool)
        or not isinstance(topology.get("controller_pid"), int)
        or topology["controller_pid"] <= 0
    ):
        errors.append("execution topology has no controller process identity")
    if invocation.comparison_mode == "performance":
        expected_scalars = {
            "controller_process_reused": True,
            "variant_worker_process_sets": 1,
            "variant_worker_process_reused": True,
            "same_gpu_worker_process": True,
            "same_cuda_stream_proven": True,
            "shared_model_instance": True,
            "reference_attention_backend_override": "fa",
            "candidate_attention_backend_override": None,
            "port_isolation": expected_ports,
        }
        if any(
            topology.get(name) != value for name, value in expected_scalars.items()
        ):
            errors.append(
                "execution topology does not match shared performance worker semantics"
            )
        if topology.get("variant_worker_lifecycle") != (
            "one long-lived candidate-configured DiffGenerator scheduler worker "
            "reused for both variants and run orders"
        ):
            errors.append("execution topology does not describe the worker lifecycle")
        worker_topology = topology.get("worker_execution_topology")
        if (
            not isinstance(worker_topology, dict)
            or isinstance(worker_topology.get("worker_pid"), bool)
            or not isinstance(worker_topology.get("worker_pid"), int)
            or worker_topology["worker_pid"] <= 0
            or not isinstance(worker_topology.get("cuda_device"), str)
            or not worker_topology["cuda_device"].startswith("cuda")
            or isinstance(worker_topology.get("cuda_stream_handle"), bool)
            or not isinstance(worker_topology.get("cuda_stream_handle"), int)
        ):
            errors.append("shared performance worker identity is missing")
        order_results = report.get("order_results")
        if isinstance(worker_topology, dict) and isinstance(order_results, dict):
            expected_measure_runs = _invocation_integer_option(
                invocation, "--measure-runs"
            )
            for run_order in QUALIFICATION_RUN_ORDERS:
                order_result = order_results.get(run_order)
                for variant in ("reference", "candidate"):
                    generation = (
                        order_result.get(f"{variant}_generation")
                        if isinstance(order_result, dict)
                        else None
                    )
                    observed = (
                        generation.get("per_run_worker_execution_topology")
                        if isinstance(generation, dict)
                        else None
                    )
                    if observed != [worker_topology] * (expected_measure_runs or 0):
                        errors.append(
                            f"{run_order}.{variant}: shared worker/stream evidence "
                            "is incomplete"
                        )
        elif not isinstance(order_results, dict):
            errors.append("shared performance order results are missing")
    else:
        expected_scalars = {
            "controller_process_reused": True,
            "variant_worker_process_sets": 2,
            "variant_worker_process_reused": False,
            "same_gpu_worker_process": False,
            "same_cuda_stream_proven": False,
            "port_isolation": expected_ports,
        }
        if any(
            topology.get(name) != value for name, value in expected_scalars.items()
        ):
            errors.append("execution topology does not match isolated worker semantics")
        if topology.get("variant_worker_lifecycle") != (
            "fresh local DiffGenerator scheduler process set per variant and run order"
        ):
            errors.append("execution topology does not describe the worker lifecycle")
    ports = [expected_ports[name] for name in expected_ports if name.endswith("port")]
    if any(port is None for port in ports) or len(set(ports)) != 4:
        errors.append("invocation does not use four distinct explicit ports")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("port_isolation") != (
        expected_ports
    ):
        errors.append("port provenance does not match the invocation")
    return errors


def validate_qualification_report(
    report: Any,
    invocation: QualificationInvocation,
    config: WanQualificationConfig,
) -> list[str]:
    if not isinstance(report, dict):
        return ["report is not a JSON object"]
    errors = []
    errors.extend(_validate_report_provenance(report, config))
    errors.extend(_validate_execution_topology(report, invocation))
    errors.extend(
        validate_reference_attention_backend_identity(
            report.get("reference_attention_backend_identity")
        )
    )
    if report.get("model_id") != config.model_id:
        errors.append("model_id does not match the qualification config")
    if report.get("prompt") != config.prompt or report.get("seed") != config.seed:
        errors.append("prompt or seed does not match the qualification config")
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
        for variant, variant_server in (
            ("reference", reference_server),
            ("candidate", candidate_server),
        ):
            if isinstance(variant_server, dict) and (
                "transformer_weights_path" in variant_server
                or "component_paths" in variant_server
            ):
                errors.append(f"{variant} model override is forbidden")
        if invocation.scenario in ("single-block", "generation") and isinstance(
            candidate_server, dict
        ):
            expected_layers = (
                WAN_HYBRID_SINGLE_BLOCK_LAYER_INDICES
                if invocation.scenario == "single-block"
                else WAN_HYBRID_PROMOTION_LAYER_INDICES
            )
            expected_config = json.dumps(
                {
                    "wan_hybrid_layer_indices": expected_layers,
                    "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if candidate_server.get("attention_backend_config") != expected_config:
                errors.append(
                    f"{invocation.scenario} candidate is not locked to the "
                    "promotion route at t521"
                )

    sampling_kwargs = report.get("sampling_kwargs")
    if not isinstance(sampling_kwargs, dict):
        errors.append("missing sampling_kwargs")
    else:
        expected_sampling = {
            "width": config.width,
            "height": config.height,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "guidance_scale_2": config.guidance_scale_2,
        }
        for field_name, expected_value in expected_sampling.items():
            if sampling_kwargs.get(field_name) != expected_value:
                errors.append(f"sampling_kwargs.{field_name} is inconsistent")
        if invocation.comparison_mode == "correctness":
            if sampling_kwargs.get("return_trajectory_latents") is not True:
                errors.append("correctness did not capture trajectory latents")
        else:
            if sampling_kwargs.get("return_trajectory_latents") is not False:
                errors.append("performance unexpectedly captured trajectory latents")
            if sampling_kwargs.get("return_trajectory_decoded") is not False:
                errors.append("performance unexpectedly captured decoded trajectory")

    if invocation.comparison_mode == "correctness":
        if isinstance(qualification, dict):
            if qualification.get("thresholds") != MODEL_QUALIFICATION_THRESHOLDS:
                errors.append("correctness thresholds do not match the protocol")
            candidate_generation = report.get("candidate_generation")
            if not isinstance(candidate_generation, dict):
                candidate_generation = {}
            errors.extend(
                _validate_candidate_hit_qualification(
                    qualification,
                    actual_hit_counts=candidate_generation.get(
                        "per_run_wan_hybrid_hit_count"
                    ),
                    expected_hit_counts=candidate_generation.get(
                        "per_run_wan_hybrid_expected_hit_count"
                    ),
                    location="qualification",
                )
            )
        errors.extend(
            _validate_generation_summary(
                report.get("reference_generation"),
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
                require_hits=False,
                location="reference_generation",
                scenario=invocation.scenario,
                expected_num_steps=config.num_inference_steps,
                expected_num_frames=config.num_frames,
                expected_frame_shape=(config.height, config.width, 3),
            )
        )
        errors.extend(
            _validate_generation_summary(
                report.get("candidate_generation"),
                expected_warmup_runs=config.warmup_runs,
                expected_measure_runs=config.measure_runs,
                require_hits=True,
                location="candidate_generation",
                scenario=invocation.scenario,
                expected_num_steps=config.num_inference_steps,
                expected_num_frames=config.num_frames,
                expected_frame_shape=(config.height, config.width, 3),
            )
        )
        cross = report.get("cross_variant_metrics")
        expected_cross_pairs = set(
            itertools.product(range(config.measure_runs), range(config.measure_runs))
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
                        scenario=invocation.scenario,
                        expected_num_steps=config.num_inference_steps,
                        expected_num_frames=config.num_frames,
                        expected_frame_shape=(config.height, config.width, 3),
                    )
                )
                errors.extend(
                    _validate_generation_summary(
                        order_result.get("candidate_generation"),
                        expected_warmup_runs=config.warmup_runs,
                        expected_measure_runs=config.measure_runs,
                        require_hits=True,
                        location=f"order_results.{run_order}.candidate_generation",
                        scenario=invocation.scenario,
                        expected_num_steps=config.num_inference_steps,
                        expected_num_frames=config.num_frames,
                        expected_frame_shape=(config.height, config.width, 3),
                    )
                )
                speedup = order_result.get("performance", {}).get("wall_median_speedup")
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
                "warmup_runs_equals": MIN_QUALIFICATION_WARMUP_RUNS,
                "measure_runs_equals": MIN_QUALIFICATION_MEASURE_RUNS,
                "speedup_min": MODEL_QUALIFICATION_THRESHOLDS["speedup_min"],
                "candidate_hit_count_equals_expected": True,
                "expected_hit_count_min_exclusive": 0,
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
    parser.add_argument("--reference-fp4-gemm-backend", default="flashinfer_trtllm")
    parser.add_argument("--candidate-fp4-gemm-backend", default="flashinfer_trtllm")
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
            "Independent single-forward correctness report; pass exactly four "
            "covering transformer and transformer_2 in both execution orders "
            "when selecting the full-transformer scenario."
        ),
    )
    parser.add_argument(
        "--full-transformer-performance-report",
        dest="full_transformer_performance_reports",
        action="append",
        type=Path,
        default=[],
        help=(
            "Independent transformer_2@t521 trajectory-off AB/BA performance "
            "report; pass exactly one when selecting full-transformer performance."
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
        full_transformer_performance_reports=tuple(
            path.expanduser().resolve()
            for path in args.full_transformer_performance_reports
        ),
    )
    manifest = run_qualification_plan(config, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
