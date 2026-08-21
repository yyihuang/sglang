"""Qualify one full Wan transformer forward with per-block coverage.

The public entry point accepts already-loaded reference and candidate model
instances plus zero-argument forward callables.  It records every module in
``model.blocks`` using forward hooks, so callers can replay a real captured Wan
input without coupling this evaluator to a particular checkpoint loader.

The same model instance is reused for every measured run.  Reference/candidate
quality uses the trajectory evaluator's fixed tolerances over every run pair
and block output; repeatability must be bitwise exact for every same-variant
run pair.  Hook capture is a correctness-only path and is not a performance
measurement.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MIN_QUALIFICATION_MEASURE_RUNS,
    MIN_QUALIFICATION_WARMUP_RUNS,
    MODEL_QUALIFICATION_THRESHOLDS,
    QUALIFICATION_RUN_ORDERS,
    compute_tensor_metrics,
    evaluate_candidate_backend_hit_qualification,
    evaluate_correctness_qualification,
    summarize_trajectory_metrics,
    validate_qualification_protocol,
)


@dataclass(frozen=True)
class WanTransformerForwardTrace:
    block_outputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    wan_hybrid_hit_count: int | None = None

    @property
    def trajectory_latents(self) -> torch.Tensor:
        return torch.stack(self.block_outputs, dim=1)

    @property
    def trajectory_timesteps(self) -> torch.Tensor:
        return torch.arange(len(self.block_outputs), dtype=torch.float32)


def _extract_tensor(value: Any, *, location: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if (
        isinstance(value, (tuple, list))
        and value
        and isinstance(value[0], torch.Tensor)
    ):
        return value[0]
    raise TypeError(f"{location} must be a tensor or begin with a tensor")


def _snapshot_tensor(value: Any, *, location: str) -> torch.Tensor:
    tensor = _extract_tensor(value, location=location)
    return tensor.detach().cpu().clone()


def capture_wan_transformer_forward(
    model: Any,
    forward_call: Callable[[], Any],
    *,
    wan_hybrid_hit_count: int | None = None,
) -> WanTransformerForwardTrace:
    """Run one forward and snapshot every transformer block output.

    ``forward_call`` should invoke ``model`` with a fixed real input capture.
    Forward hooks are always removed, including when the model raises.
    """

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("Wan transformer model must expose a blocks collection")
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("Wan transformer model contains no blocks")

    block_outputs: list[torch.Tensor] = []

    def record_block_output(_module, _inputs, output) -> None:
        block_outputs.append(
            _snapshot_tensor(output, location=f"block {len(block_outputs)} output")
        )

    hooks = [block.register_forward_hook(record_block_output) for block in blocks]
    try:
        with torch.inference_mode():
            output = forward_call()
    finally:
        for hook in hooks:
            hook.remove()

    if len(block_outputs) != len(blocks):
        raise RuntimeError(
            "Full-transformer forward did not execute every block: "
            f"captured {len(block_outputs)} of {len(blocks)}"
        )
    return WanTransformerForwardTrace(
        block_outputs=tuple(block_outputs),
        output=_snapshot_tensor(output, location="transformer output"),
        wan_hybrid_hit_count=wan_hybrid_hit_count,
    )


def _summarize_trace_pair(
    reference: WanTransformerForwardTrace,
    candidate: WanTransformerForwardTrace,
    *,
    reference_run_index: int,
    candidate_run_index: int,
) -> dict[str, Any]:
    if len(reference.block_outputs) != len(candidate.block_outputs):
        raise ValueError(
            "Reference and candidate executed different transformer block counts: "
            f"{len(reference.block_outputs)} vs {len(candidate.block_outputs)}"
        )
    return {
        "reference_run_index": reference_run_index,
        "candidate_run_index": candidate_run_index,
        "trajectory_metrics": summarize_trajectory_metrics(
            reference.trajectory_latents,
            candidate.trajectory_latents,
            reference_timesteps=reference.trajectory_timesteps,
            candidate_timesteps=candidate.trajectory_timesteps,
            step_index=-1,
        ),
        # Keep the established output_metrics shape so the shared correctness
        # evaluator can enforce final-output finiteness and repeatability.
        "output_metrics": {
            "all_frames_metrics": compute_tensor_metrics(
                reference.output, candidate.output
            )
        },
    }


def _comparison_envelope(comparisons: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not comparisons:
        raise ValueError("At least one transformer comparison is required")
    all_steps = [
        metrics
        for comparison in comparisons
        for metrics in comparison["trajectory_metrics"]["per_step_metrics"]
    ]
    outputs = [
        comparison["output_metrics"]["all_frames_metrics"]
        for comparison in comparisons
    ]
    return {
        "min_all_blocks_cosine": min(
            metrics["cosine_similarity"] for metrics in all_steps
        ),
        "max_all_blocks_mae": max(metrics["mae"] for metrics in all_steps),
        "max_all_blocks_max_abs": max(metrics["max_abs"] for metrics in all_steps),
        "min_output_cosine": min(
            metrics["cosine_similarity"] for metrics in outputs
        ),
        "max_output_mae": max(metrics["mae"] for metrics in outputs),
        "max_output_max_abs": max(metrics["max_abs"] for metrics in outputs),
    }


def summarize_transformer_forward_repeatability(
    traces: Sequence[WanTransformerForwardTrace],
) -> dict[str, Any]:
    num_runs = len(traces)
    if num_runs < 2:
        return {
            "available": False,
            "num_runs": num_runs,
            "reason": "repeatability requires at least two measured runs",
        }
    comparisons = [
        _summarize_trace_pair(
            traces[reference_run_index],
            traces[candidate_run_index],
            reference_run_index=reference_run_index,
            candidate_run_index=candidate_run_index,
        )
        for reference_run_index, candidate_run_index in itertools.combinations(
            range(num_runs), 2
        )
    ]
    return {
        "available": True,
        "num_runs": num_runs,
        "pairing": "all-pairs",
        "num_pairs": len(comparisons),
        "comparisons": comparisons,
        "envelope": _comparison_envelope(comparisons),
    }


def summarize_transformer_forward_cross_variant(
    reference_traces: Sequence[WanTransformerForwardTrace],
    candidate_traces: Sequence[WanTransformerForwardTrace],
) -> dict[str, Any]:
    if not reference_traces or not candidate_traces:
        raise ValueError("Cross-variant metrics require both measured trace sets")
    comparisons = [
        _summarize_trace_pair(
            reference,
            candidate,
            reference_run_index=reference_run_index,
            candidate_run_index=candidate_run_index,
        )
        for reference_run_index, reference in enumerate(reference_traces)
        for candidate_run_index, candidate in enumerate(candidate_traces)
    ]
    return {
        "reference_num_runs": len(reference_traces),
        "candidate_num_runs": len(candidate_traces),
        "pairing": "cross-product",
        "num_pairs": len(comparisons),
        "comparisons": comparisons,
        "envelope": _comparison_envelope(comparisons),
    }


def _cross_variant_output_quality_failures(
    comparisons: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for comparison in comparisons:
        metrics = comparison["output_metrics"]["all_frames_metrics"]
        location = {
            "scope": "cross_variant",
            "reference_run_index": comparison["reference_run_index"],
            "candidate_run_index": comparison["candidate_run_index"],
            "location": "transformer_output",
        }
        if not metrics["finite"]:
            continue
        if not metrics["within_tolerance"]:
            failures.append(
                location
                | {
                    "reason": "outside_atol_rtol",
                    "max_abs": metrics["max_abs"],
                }
            )
        if metrics["cosine_similarity"] < MODEL_QUALIFICATION_THRESHOLDS[
            "cosine_min"
        ]:
            failures.append(
                location
                | {
                    "reason": "cosine_below_minimum",
                    "cosine_similarity": metrics["cosine_similarity"],
                }
            )
        if metrics["mae"] > MODEL_QUALIFICATION_THRESHOLDS["mae_max"]:
            failures.append(
                location | {"reason": "mae_above_maximum", "mae": metrics["mae"]}
            )
    return failures


def evaluate_transformer_forward_correctness(
    reference_traces: Sequence[WanTransformerForwardTrace],
    candidate_traces: Sequence[WanTransformerForwardTrace],
) -> dict[str, Any]:
    cross_variant = summarize_transformer_forward_cross_variant(
        reference_traces, candidate_traces
    )
    repeatability = {
        "reference": summarize_transformer_forward_repeatability(reference_traces),
        "candidate": summarize_transformer_forward_repeatability(candidate_traces),
    }
    qualification = evaluate_correctness_qualification(
        cross_variant, repeatability
    )
    output_failures = _cross_variant_output_quality_failures(
        cross_variant["comparisons"]
    )
    qualification = qualification | {
        "passed": qualification["passed"] and not output_failures,
        "failures": qualification["failures"] + output_failures,
    }
    hit_qualification = evaluate_candidate_backend_hit_qualification(
        [trace.wan_hybrid_hit_count for trace in candidate_traces],
    )
    qualification = qualification | {
        "passed": qualification["passed"] and hit_qualification["passed"],
        "failures": qualification["failures"]
        + [
            {"scope": "candidate_backend_hits"} | failure
            for failure in hit_qualification["failures"]
        ],
        "candidate_backend_hits": hit_qualification,
    }
    return {
        "cross_variant_metrics": cross_variant,
        "repeatability": repeatability,
        "qualification": qualification,
    }


def _run_variant(
    model: Any,
    forward_call: Callable[[], Any],
    *,
    warmup_runs: int,
    measure_runs: int,
    reset_hit_count: Callable[[], None] | None = None,
    read_hit_count: Callable[[], int] | None = None,
) -> list[WanTransformerForwardTrace]:
    if (reset_hit_count is None) != (read_hit_count is None):
        raise ValueError("reset_hit_count and read_hit_count must be provided together")
    for _ in range(warmup_runs):
        with torch.inference_mode():
            forward_call()

    traces = []
    for _ in range(measure_runs):
        if reset_hit_count is not None:
            reset_hit_count()
        trace = capture_wan_transformer_forward(model, forward_call)
        if read_hit_count is not None:
            hit_count = read_hit_count()
            if isinstance(hit_count, bool) or not isinstance(hit_count, int):
                hit_count = None
            trace = WanTransformerForwardTrace(
                block_outputs=trace.block_outputs,
                output=trace.output,
                wan_hybrid_hit_count=hit_count,
            )
        traces.append(trace)
    return traces


def run_wan_transformer_forward_qualification(
    *,
    reference_model: Any,
    candidate_model: Any,
    reference_forward: Callable[[], Any],
    candidate_forward: Callable[[], Any],
    reset_candidate_hit_count: Callable[[], None],
    read_candidate_hit_count: Callable[[], int],
    run_order: str = "reference-first",
    warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
) -> dict[str, Any]:
    """Run a full-transformer correctness qualification on fixed real inputs."""

    validate_qualification_protocol(
        comparison_mode="correctness",
        run_order=run_order,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    )
    variant_calls = {
        "reference": lambda: _run_variant(
            reference_model,
            reference_forward,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
        ),
        "candidate": lambda: _run_variant(
            candidate_model,
            candidate_forward,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            reset_hit_count=reset_candidate_hit_count,
            read_hit_count=read_candidate_hit_count,
        ),
    }
    execution_order = (
        ("reference", "candidate")
        if run_order == "reference-first"
        else ("candidate", "reference")
    )
    traces: dict[str, list[WanTransformerForwardTrace]] = {}
    for variant in execution_order:
        traces[variant] = variant_calls[variant]()

    result = evaluate_transformer_forward_correctness(
        traces["reference"], traces["candidate"]
    )
    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "num_blocks": len(traces["reference"][0].block_outputs),
        "candidate_per_run_wan_hybrid_hit_count": [
            trace.wan_hybrid_hit_count for trace in traces["candidate"]
        ],
        **result,
    }


def validate_wan_transformer_forward_report(
    report: Any,
    *,
    expected_warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    expected_measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
) -> list[str]:
    """Validate a serialized real-input full-transformer report."""

    if not isinstance(report, dict):
        return ["report is not a JSON object"]
    errors = []
    if report.get("comparison_mode") != "correctness":
        errors.append("comparison_mode is not correctness")
    if report.get("run_order") not in QUALIFICATION_RUN_ORDERS:
        errors.append("run_order is not an explicit qualification order")
    if report.get("warmup_runs") != expected_warmup_runs:
        errors.append("warmup run count does not match the protocol")
    if report.get("measure_runs") != expected_measure_runs:
        errors.append("measured run count does not match the protocol")
    num_blocks = report.get("num_blocks")
    if (
        isinstance(num_blocks, bool)
        or not isinstance(num_blocks, int)
        or num_blocks <= 0
    ):
        errors.append("num_blocks must be a positive integer")

    expected_cross_pairs = set(
        itertools.product(
            range(expected_measure_runs), range(expected_measure_runs)
        )
    )
    expected_repeat_pairs = set(
        itertools.combinations(range(expected_measure_runs), 2)
    )

    def validate_comparisons(
        comparisons: Any,
        *,
        expected_pairs: set[tuple[int, int]],
        location: str,
    ) -> None:
        if not isinstance(comparisons, list):
            errors.append(f"{location}: missing comparisons")
            return
        actual_pairs = {
            (
                comparison.get("reference_run_index"),
                comparison.get("candidate_run_index"),
            )
            for comparison in comparisons
            if isinstance(comparison, dict)
        }
        if actual_pairs != expected_pairs or len(comparisons) != len(expected_pairs):
            errors.append(f"{location}: run-pair coverage is incomplete")
        for pair_index, comparison in enumerate(comparisons):
            if not isinstance(comparison, dict):
                errors.append(f"{location}[{pair_index}]: invalid comparison")
                continue
            trajectory = comparison.get("trajectory_metrics")
            per_step = (
                trajectory.get("per_step_metrics")
                if isinstance(trajectory, dict)
                else None
            )
            num_steps = (
                trajectory.get("num_steps")
                if isinstance(trajectory, dict)
                else None
            )
            if (
                not isinstance(per_step, list)
                or not isinstance(num_blocks, int)
                or num_steps != num_blocks
                or len(per_step) != num_blocks
            ):
                errors.append(f"{location}[{pair_index}]: block coverage is incomplete")

    cross_variant = report.get("cross_variant_metrics")
    if (
        not isinstance(cross_variant, dict)
        or cross_variant.get("pairing") != "cross-product"
    ):
        errors.append("cross-variant summary is not a cross-product")
    else:
        validate_comparisons(
            cross_variant.get("comparisons"),
            expected_pairs=expected_cross_pairs,
            location="cross_variant_metrics.comparisons",
        )

    repeatability = report.get("repeatability")
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
            validate_comparisons(
                summary.get("comparisons"),
                expected_pairs=expected_repeat_pairs,
                location=f"{location}.comparisons",
            )

    hit_counts = report.get("candidate_per_run_wan_hybrid_hit_count")
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
        errors.append("every measured candidate hit count must be positive")

    qualification = report.get("qualification")
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("failures") != []
        or qualification.get("thresholds") != MODEL_QUALIFICATION_THRESHOLDS
    ):
        errors.append("correctness qualification is incomplete")
    elif qualification.get("candidate_backend_hits") != {
        "passed": True,
        "thresholds": {"candidate_hit_count_min_exclusive": 0},
        "failures": [],
    }:
        errors.append("candidate backend-hit qualification is incomplete")
    return errors


def write_wan_transformer_forward_report(
    report: dict[str, Any], output_path: str | Path
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
