"""Qualify one captured Wan transformer forward with per-block coverage.

The public entry point accepts already-loaded reference and candidate model
instances plus a verified worker-produced input manifest. It invokes both
models itself and records every module in ``model.blocks`` using forward hooks,
so caller-owned mappings or closures cannot substitute an unreported input.

The same model instance is reused for every measured run.  Reference/candidate
quality uses the trajectory evaluator's fixed tolerances over every run pair
and block output; repeatability must be bitwise exact for every same-variant
run pair.  Hook capture is a correctness-only path and is not a performance
measurement.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import time
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    WanHybridEvidenceCollector,
    validate_wan_hybrid_exact_serving_boundary_evidence,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import (
    LocalAttention,
    USPAttention,
    UlyssesAttention,
    apply_attention_backend_override,
    prepare_attention_backend_override,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.qualification.attention_backend_identity import (
    collect_runtime_attention_backend_identity,
)
from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    _model_identity,
    load_wan_transformer_input_capture,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MIN_QUALIFICATION_MEASURE_RUNS,
    MIN_QUALIFICATION_WARMUP_RUNS,
    MODEL_QUALIFICATION_THRESHOLDS,
    QUALIFICATION_RUN_ORDERS,
    WAN_HYBRID_PROMOTION_LAYER_INDICES,
    WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
    _collapse_reference_attention_backend_identities,
    compute_tensor_metrics,
    evaluate_candidate_backend_hit_qualification,
    evaluate_correctness_qualification,
    summarize_trajectory_metrics,
    validate_qualification_protocol,
    validate_reference_attention_backend_identity,
)


@dataclass(frozen=True)
class WanTransformerForwardTrace:
    block_outputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    request_id: str
    component_name: str
    step_index: int
    actual_timestep: int
    cfg_branch_index: int
    wan_hybrid_coverage: dict[str, Any]

    @property
    def wan_hybrid_hit_count(self) -> int:
        return self.wan_hybrid_coverage["actual_hit_count"]

    @property
    def trajectory_latents(self) -> torch.Tensor:
        return torch.stack(self.block_outputs, dim=1)

    @property
    def trajectory_timesteps(self) -> torch.Tensor:
        return torch.arange(len(self.block_outputs), dtype=torch.float32)


@dataclass(frozen=True)
class WanTransformerDirectRequest:
    """Minimal request identity carried by a direct qualification context."""

    request_id: str
    enable_sequence_shard: bool = False
    enable_teacache: bool = False
    enable_spectrum: bool = False


@dataclass(frozen=True)
class WanTransformerForwardTiming:
    duration_ms: float
    request_id: str
    component_name: str
    step_index: int
    actual_timestep: int
    cfg_branch_index: int
    num_blocks: int
    controller_pid: int
    cuda_stream_handle: int
    wan_hybrid_coverage: dict[str, Any]
    output_summary: dict[str, Any]
    attention_backend_identity: dict[str, Any] | None = None

    @property
    def wan_hybrid_hit_count(self) -> int:
        return self.wan_hybrid_coverage["actual_hit_count"]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_fixed_input(value: Any, *, location: str = "input") -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{location} must use strided tensor storage")
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{location} mapping keys must be strings")
        return {
            key: _summarize_fixed_input(item, location=f"{location}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [
            _summarize_fixed_input(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{location} has unsupported provenance type {type(value).__name__}"
    )


def build_wan_transformer_evidence_binding(
    *,
    component_name: str,
    component_model_path: str | Path,
    fixed_input: Any,
    reference_model: Any,
    candidate_model: Any,
) -> dict[str, Any]:
    """Bind a report to the loaded component path, model objects, and inputs."""

    if component_name not in ("transformer", "transformer_2"):
        raise ValueError("component_name must be transformer or transformer_2")
    component_path = Path(component_model_path).expanduser().resolve()
    config_path = component_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"component config is required for provenance: {config_path}"
        )
    config_files = [{"path": "config.json", "sha256": _sha256_file(config_path)}]
    index_path = component_path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        config_files.append(
            {"path": index_path.name, "sha256": _sha256_file(index_path)}
        )
    input_summary = _summarize_fixed_input(fixed_input)
    binding = {
        "schema_version": 1,
        "component_name": component_name,
        "resolved_component_path": str(component_path),
        "component_config_files": config_files,
        "fixed_input": input_summary,
        "fixed_input_sha256": _sha256_json(input_summary),
        "reference_model": _model_identity(reference_model),
        "candidate_model": _model_identity(candidate_model),
    }
    return binding | {"binding_sha256": _sha256_json(binding)}


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


def _wan_transformer_autocast_context(
    fixed_input: Mapping[str, Any],
) -> AbstractContextManager:
    """Replay the production BF16 denoising autocast for captured CUDA input."""

    hidden_states = fixed_input.get("hidden_states")
    enabled = (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
    )
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=enabled,
    )


def capture_wan_transformer_forward(
    model: Any,
    *,
    fixed_input: Mapping[str, Any],
    request_id: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
) -> WanTransformerForwardTrace:
    """Directly run one request-bound forward and snapshot every block output."""

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
    collector = WanHybridEvidenceCollector(request_id=request_id)
    try:
        with (
            torch.inference_mode(),
            _wan_transformer_autocast_context(fixed_input),
            set_forward_context(
                current_timestep=step_index,
                attn_metadata=None,
                forward_batch=WanTransformerDirectRequest(request_id=request_id),
                wan_component_name=component_name,
                wan_actual_timestep=actual_timestep,
                wan_cfg_branch_index=cfg_branch_index,
                wan_hybrid_evidence_collector=collector,
            ),
        ):
            output = model(**fixed_input)
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
        request_id=request_id,
        component_name=component_name,
        step_index=step_index,
        actual_timestep=actual_timestep,
        cfg_branch_index=cfg_branch_index,
        wan_hybrid_coverage=collector.coverage(),
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
        comparison["output_metrics"]["all_frames_metrics"] for comparison in comparisons
    ]
    return {
        "min_all_blocks_cosine": min(
            metrics["cosine_similarity"] for metrics in all_steps
        ),
        "max_all_blocks_mae": max(metrics["mae"] for metrics in all_steps),
        "max_all_blocks_max_abs": max(metrics["max_abs"] for metrics in all_steps),
        "min_output_cosine": min(metrics["cosine_similarity"] for metrics in outputs),
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
        if metrics["cosine_similarity"] < MODEL_QUALIFICATION_THRESHOLDS["cosine_min"]:
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


def _direct_coverage_failure(
    trace: Any,
    *,
    variant: str,
    candidate_layer_indices: Sequence[int],
    num_blocks: int | None = None,
) -> dict[str, Any] | None:
    coverage = trace.wan_hybrid_coverage
    expected_layers = list(
        range(num_blocks if num_blocks is not None else len(trace.block_outputs))
    )
    candidate = variant == "candidate"
    expected_hybrid_layers = list(candidate_layer_indices) if candidate else []
    expected_hits = len(expected_hybrid_layers)
    expected_successes = expected_hits
    if coverage.get("request_id") != trace.request_id:
        return {"variant": variant, "reason": "request_id_mismatch"}
    expected_scalars = {
        "schema_version": 2,
        "expected_hit_count": expected_hits,
        "actual_hit_count": expected_hits,
        "attributed_actual_hit_count": expected_hits,
        "unattributed_actual_hit_count": 0,
        "eligible_hybrid_miss_count": 0,
        "num_route_events": len(expected_layers),
        "num_success_events": expected_successes,
    }
    if any(coverage.get(name) != value for name, value in expected_scalars.items()):
        return {
            "variant": variant,
            "reason": "coverage_scalar_mismatch",
            "expected": expected_scalars,
            "actual": {name: coverage.get(name) for name in expected_scalars},
        }
    boundary_errors = validate_wan_hybrid_exact_serving_boundary_evidence(coverage)
    if boundary_errors:
        return {
            "variant": variant,
            "reason": "serving_boundary_evidence_mismatch",
            "errors": boundary_errors,
        }
    steps = coverage.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        return {"variant": variant, "reason": "coverage_step_mismatch"}
    step = steps[0]
    expected_step = {
        "step_index": trace.step_index,
        "actual_timestep": trace.actual_timestep,
        "active_component": trace.component_name,
        "executed_cfg_branch_indices": [trace.cfg_branch_index],
    }
    if not isinstance(step, dict) or any(
        step.get(name) != value for name, value in expected_step.items()
    ):
        return {
            "variant": variant,
            "reason": "coverage_coordinate_mismatch",
            "expected": expected_step,
            "actual": (
                {name: step.get(name) for name in expected_step}
                if isinstance(step, dict)
                else step
            ),
        }
    branches = step.get("branches") if isinstance(step, dict) else None
    if not isinstance(branches, list) or len(branches) != 1:
        return {"variant": variant, "reason": "coverage_branch_mismatch"}
    branch = branches[0]
    expected_branch = {
        "cfg_branch_index": trace.cfg_branch_index,
        "num_layers": len(expected_layers),
        "layer_indices": expected_layers,
        "eligible_layer_indices": expected_hybrid_layers,
        "planned_hybrid_layer_indices": expected_hybrid_layers,
        "successful_hybrid_layer_indices": expected_hybrid_layers,
        "eligible_hybrid_miss_layer_indices": [],
        "unexpected_successful_hybrid_layer_indices": [],
        "configured_fallback_layer_indices": (
            [index for index in expected_layers if index not in expected_hybrid_layers]
            if candidate
            else []
        ),
        "control_layer_indices": [] if candidate else expected_layers,
        "expected_hit_count": expected_hits,
        "actual_hit_count": expected_hits,
    }
    if any(branch.get(name) != value for name, value in expected_branch.items()):
        return {
            "variant": variant,
            "reason": "coverage_route_mismatch",
            "expected": expected_branch,
            "actual": {name: branch.get(name) for name in expected_branch},
        }
    return None


def _prepare_shared_performance_model(model: Any) -> list[Any]:
    """Prepare FA on one candidate-configured model without changing its default."""

    layers = [
        module
        for module in model.modules()
        if isinstance(module, (LocalAttention, UlyssesAttention, USPAttention))
    ]
    if not layers:
        raise ValueError("performance model exposes no switchable attention layers")
    for layer in layers:
        prepare_attention_backend_override(layer, AttentionBackendEnum.FA)
    return layers


def _select_shared_performance_variant(layers: Sequence[Any], variant: str) -> None:
    if variant not in ("reference", "candidate"):
        raise ValueError(f"unknown performance variant: {variant}")
    target = AttentionBackendEnum.FA if variant == "reference" else None
    for layer in layers:
        apply_attention_backend_override(layer, target)


def _warmup_wan_transformer_forward(
    model: Any,
    *,
    fixed_input: Mapping[str, Any],
    request_id: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
) -> None:
    """Warm up the exact request path without retaining multi-GB hook snapshots."""

    collector = WanHybridEvidenceCollector(request_id=request_id)
    with (
        torch.inference_mode(),
        _wan_transformer_autocast_context(fixed_input),
        set_forward_context(
            current_timestep=step_index,
            attn_metadata=None,
            forward_batch=WanTransformerDirectRequest(request_id=request_id),
            wan_component_name=component_name,
            wan_actual_timestep=actual_timestep,
            wan_cfg_branch_index=cfg_branch_index,
            wan_hybrid_evidence_collector=collector,
        ),
    ):
        model(**fixed_input)


def _timed_wan_transformer_forward(
    model: Any,
    *,
    fixed_input: Mapping[str, Any],
    request_id: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
    device: torch.device,
    requested_attention_backend: str | None,
) -> WanTransformerForwardTiming:
    """Time one synchronized forward without hooks or trajectory snapshots."""

    collector = WanHybridEvidenceCollector(request_id=request_id)
    controller_pid = os.getpid()
    cuda_stream_handle = int(torch.cuda.current_stream(device).cuda_stream)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        _wan_transformer_autocast_context(fixed_input),
        set_forward_context(
            current_timestep=step_index,
            attn_metadata=None,
            forward_batch=WanTransformerDirectRequest(request_id=request_id),
            wan_component_name=component_name,
            wan_actual_timestep=actual_timestep,
            wan_cfg_branch_index=cfg_branch_index,
            wan_hybrid_evidence_collector=collector,
        ),
    ):
        output = model(**fixed_input)
    torch.cuda.synchronize(device)
    if int(torch.cuda.current_stream(device).cuda_stream) != cuda_stream_handle:
        raise RuntimeError("current CUDA stream changed during direct forward timing")
    duration_ms = (time.perf_counter() - started) * 1000.0
    output_tensor = _extract_tensor(output, location="transformer output")
    output_summary = {
        "shape": list(output_tensor.shape),
        "dtype": str(output_tensor.dtype),
        "device": str(output_tensor.device),
        "finite": bool(torch.isfinite(output_tensor).all().item()),
    }
    attention_backend_identity = collect_runtime_attention_backend_identity(
        [model], requested_backend=requested_attention_backend
    )
    return WanTransformerForwardTiming(
        duration_ms=duration_ms,
        request_id=request_id,
        component_name=component_name,
        step_index=step_index,
        actual_timestep=actual_timestep,
        cfg_branch_index=cfg_branch_index,
        num_blocks=len(model.blocks),
        controller_pid=controller_pid,
        cuda_stream_handle=cuda_stream_handle,
        wan_hybrid_coverage=collector.coverage(),
        output_summary=output_summary,
        attention_backend_identity=attention_backend_identity,
    )


def _run_timed_variant(
    model: Any,
    *,
    fixed_input: Mapping[str, Any],
    variant: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
    warmup_runs: int,
    measure_runs: int,
    device: torch.device,
) -> list[WanTransformerForwardTiming]:
    server_args = getattr(model, "_wan_qualification_server_args", None)
    if server_args is not None:
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        set_global_server_args(server_args)
    for _ in range(warmup_runs):
        _warmup_wan_transformer_forward(
            model,
            fixed_input=fixed_input,
            request_id=f"wan-transformer-{variant}-perf-warmup-{uuid.uuid4()}",
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
        )
    return [
        _timed_wan_transformer_forward(
            model,
            fixed_input=fixed_input,
            request_id=f"wan-transformer-{variant}-perf-measure-{uuid.uuid4()}",
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
            device=device,
            requested_attention_backend=("fa" if variant == "reference" else None),
        )
        for _ in range(measure_runs)
    ]


def evaluate_transformer_forward_correctness(
    reference_traces: Sequence[WanTransformerForwardTrace],
    candidate_traces: Sequence[WanTransformerForwardTrace],
    *,
    candidate_layer_indices: Sequence[int],
    candidate_backend_expectation: str = "exercised",
) -> dict[str, Any]:
    cross_variant = summarize_transformer_forward_cross_variant(
        reference_traces, candidate_traces
    )
    repeatability = {
        "reference": summarize_transformer_forward_repeatability(reference_traces),
        "candidate": summarize_transformer_forward_repeatability(candidate_traces),
    }
    qualification = evaluate_correctness_qualification(cross_variant, repeatability)
    output_failures = _cross_variant_output_quality_failures(
        cross_variant["comparisons"]
    )
    qualification = qualification | {
        "passed": qualification["passed"] and not output_failures,
        "failures": qualification["failures"] + output_failures,
    }
    hit_counts = [trace.wan_hybrid_hit_count for trace in candidate_traces]
    expected_hit_counts = [len(candidate_layer_indices)] * len(candidate_traces)
    if candidate_backend_expectation not in ("exercised", "temporal_fallback"):
        raise ValueError("unsupported candidate backend expectation")
    if candidate_backend_expectation == "exercised":
        hit_qualification = evaluate_candidate_backend_hit_qualification(
            hit_counts,
            expected_hit_counts,
        ) | {"candidate_backend_exercised": True}
    else:
        expected_hit_counts = [0] * len(candidate_traces)
        zero_hit_failures = [
            {
                "reason": "temporal_fallback_hit_count_mismatch",
                "run_index": run_index,
                "expected_hit_count": 0,
                "actual_hit_count": hit_count,
            }
            for run_index, hit_count in enumerate(hit_counts)
            if isinstance(hit_count, bool)
            or not isinstance(hit_count, int)
            or hit_count != 0
        ]
        if candidate_layer_indices:
            zero_hit_failures.append(
                {
                    "reason": "temporal_fallback_has_eligible_layers",
                    "eligible_layer_indices": list(candidate_layer_indices),
                }
            )
        hit_qualification = {
            "passed": not zero_hit_failures,
            "thresholds": {
                "candidate_hit_count_equals_expected": True,
                "expected_hit_count_equals": 0,
            },
            "expected_hit_counts": expected_hit_counts,
            "actual_hit_counts": hit_counts,
            "failures": zero_hit_failures,
            "candidate_backend_exercised": False,
        }
    qualification = qualification | {
        "passed": qualification["passed"] and hit_qualification["passed"],
        "failures": qualification["failures"]
        + [
            {"scope": "candidate_backend_hits"} | failure
            for failure in hit_qualification["failures"]
        ],
        "candidate_backend_hits": hit_qualification,
    }
    coverage_failures = [
        failure
        for variant, variant_traces in (
            ("reference", reference_traces),
            ("candidate", candidate_traces),
        )
        for trace in variant_traces
        if (
            (
                failure := _direct_coverage_failure(
                    trace,
                    variant=variant,
                    candidate_layer_indices=candidate_layer_indices,
                )
            )
            is not None
        )
    ]
    qualification = qualification | {
        "passed": qualification["passed"] and not coverage_failures,
        "failures": qualification["failures"]
        + [
            {"scope": "request_local_backend_coverage"} | failure
            for failure in coverage_failures
        ],
        "request_local_backend_coverage": {
            "passed": not coverage_failures,
            "failures": coverage_failures,
        },
    }
    return {
        "cross_variant_metrics": cross_variant,
        "repeatability": repeatability,
        "qualification": qualification,
    }


def _eligible_wan_hybrid_layer_indices(
    model: Any,
    *,
    configured_layer_indices: Sequence[int],
    actual_timestep: int,
) -> list[int]:
    min_timestep = getattr(model, "wan_hybrid_min_timestep", None)
    max_timestep = getattr(model, "wan_hybrid_max_timestep", None)
    if min_timestep is not None and actual_timestep < min_timestep:
        return []
    if max_timestep is not None and actual_timestep > max_timestep:
        return []
    return list(configured_layer_indices)


def _run_variant(
    model: Any,
    *,
    fixed_input: Mapping[str, Any],
    variant: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
    warmup_runs: int,
    measure_runs: int,
) -> list[WanTransformerForwardTrace]:
    server_args = getattr(model, "_wan_qualification_server_args", None)
    if server_args is not None:
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        set_global_server_args(server_args)
    for _ in range(warmup_runs):
        _warmup_wan_transformer_forward(
            model,
            fixed_input=fixed_input,
            request_id=f"wan-transformer-{variant}-warmup-{uuid.uuid4()}",
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
        )

    traces = []
    for _ in range(measure_runs):
        traces.append(
            capture_wan_transformer_forward(
                model,
                fixed_input=fixed_input,
                request_id=f"wan-transformer-{variant}-measure-{uuid.uuid4()}",
                component_name=component_name,
                step_index=step_index,
                actual_timestep=actual_timestep,
                cfg_branch_index=cfg_branch_index,
            )
        )
    return traces


def _move_fixed_input_to_model(
    value: Any, *, device: torch.device, location: str = "input"
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {
            key: _move_fixed_input_to_model(
                item, device=device, location=f"{location}.{key}"
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _move_fixed_input_to_model(
                item, device=device, location=f"{location}[{index}]"
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _move_fixed_input_to_model(
                item, device=device, location=f"{location}[{index}]"
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{location} has unsupported direct-forward input type")


def _model_device(model: Any) -> torch.device:
    if is_layerwise_offloaded_module(model):
        return get_local_torch_device()
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        return tensor.device
    raise ValueError("Wan transformer model has neither parameters nor buffers")


def _validate_loaded_model_against_capture(
    model: Any, capture_model: Any, *, variant: str
) -> None:
    identity = _model_identity(model)
    if not isinstance(capture_model, dict):
        raise ValueError("Wan capture model identity is missing")
    for name in (
        "class",
        "num_blocks",
        "config_sha256",
        "parameter_manifest_sha256",
        "parameter_count",
    ):
        if identity.get(name) != capture_model.get(name):
            raise ValueError(
                f"{variant} loaded model does not match captured model field {name}"
            )


def run_wan_transformer_forward_qualification(
    *,
    reference_model: Any,
    candidate_model: Any,
    capture_manifest_path: str | Path,
    run_order: str = "reference-first",
    warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
    candidate_backend_expectation: str = "exercised",
) -> dict[str, Any]:
    """Run a full-transformer correctness qualification on fixed real inputs."""

    validate_qualification_protocol(
        comparison_mode="correctness",
        run_order=run_order,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    )
    fixed_input_cpu, capture_manifest = load_wan_transformer_input_capture(
        capture_manifest_path
    )
    component = capture_manifest.get("component")
    capture_coordinates = capture_manifest.get("capture")
    if not isinstance(component, dict) or not isinstance(capture_coordinates, dict):
        raise ValueError("Wan capture component or coordinates are missing")
    component_name = component.get("name")
    if component_name not in ("transformer", "transformer_2"):
        raise ValueError("captured component must be transformer or transformer_2")
    component_model_path = component.get("resolved_path")
    if not isinstance(component_model_path, str) or not component_model_path:
        raise ValueError("captured component path is missing")
    step_index = capture_coordinates.get("step_index")
    actual_timestep = capture_coordinates.get("actual_timestep")
    cfg_branch_index = capture_coordinates.get("cfg_branch_index")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (step_index, actual_timestep, cfg_branch_index)
    ):
        raise ValueError("captured step, timestep, and CFG branch must be integers")
    if any(not isinstance(key, str) for key in fixed_input_cpu):
        raise TypeError("captured input keys must be model-forward keyword names")
    _validate_loaded_model_against_capture(
        reference_model, capture_manifest.get("model"), variant="reference"
    )
    _validate_loaded_model_against_capture(
        candidate_model, capture_manifest.get("model"), variant="candidate"
    )
    reference_input = _move_fixed_input_to_model(
        fixed_input_cpu, device=_model_device(reference_model)
    )
    candidate_input = _move_fixed_input_to_model(
        fixed_input_cpu, device=_model_device(candidate_model)
    )
    evidence_binding = build_wan_transformer_evidence_binding(
        component_name=component_name,
        component_model_path=component_model_path,
        fixed_input=fixed_input_cpu,
        reference_model=reference_model,
        candidate_model=candidate_model,
    )
    evidence_binding = evidence_binding | {
        "capture_manifest_sha256": capture_manifest["manifest_sha256"],
        "capture_input_sha256": capture_manifest["input_sha256"],
        "capture_request_id": capture_manifest["request_id"],
        "capture_coordinates": dict(capture_coordinates),
        "capture_sampling_sha256": capture_manifest["sampling"]["sampling_sha256"],
    }
    evidence_binding["binding_sha256"] = _sha256_json(
        {
            key: value
            for key, value in evidence_binding.items()
            if key != "binding_sha256"
        }
    )
    variant_calls = {
        "reference": lambda: _run_variant(
            reference_model,
            fixed_input=reference_input,
            variant="reference",
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
        ),
        "candidate": lambda: _run_variant(
            candidate_model,
            fixed_input=candidate_input,
            variant="candidate",
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
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
    invocation_input_sha256 = _sha256_json(_summarize_fixed_input(fixed_input_cpu))
    if evidence_binding["fixed_input_sha256"] != invocation_input_sha256:
        raise RuntimeError("full-transformer forward mutated the fixed input")

    candidate_layer_indices_value = getattr(
        candidate_model, "wan_hybrid_layer_indices", None
    )
    configured_candidate_layer_indices = (
        list(range(len(traces["candidate"][0].block_outputs)))
        if candidate_layer_indices_value is None
        else sorted(candidate_layer_indices_value)
    )
    eligible_candidate_layer_indices = _eligible_wan_hybrid_layer_indices(
        candidate_model,
        configured_layer_indices=configured_candidate_layer_indices,
        actual_timestep=actual_timestep,
    )
    result = evaluate_transformer_forward_correctness(
        traces["reference"],
        traces["candidate"],
        candidate_layer_indices=eligible_candidate_layer_indices,
        candidate_backend_expectation=candidate_backend_expectation,
    )
    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "component_name": component_name,
        "capture_manifest_path": str(Path(capture_manifest_path).resolve()),
        "evidence_binding": evidence_binding,
        "invocation_input_sha256": invocation_input_sha256,
        "num_blocks": len(traces["reference"][0].block_outputs),
        "candidate_wan_hybrid_layer_indices": configured_candidate_layer_indices,
        "candidate_wan_hybrid_eligible_layer_indices": (
            eligible_candidate_layer_indices
        ),
        "candidate_backend_exercised": bool(eligible_candidate_layer_indices),
        "candidate_backend_expectation": candidate_backend_expectation,
        "candidate_wan_hybrid_min_timestep": getattr(
            candidate_model, "wan_hybrid_min_timestep", None
        ),
        "candidate_wan_hybrid_max_timestep": getattr(
            candidate_model, "wan_hybrid_max_timestep", None
        ),
        "candidate_per_run_wan_hybrid_expected_hit_count": [
            trace.wan_hybrid_coverage["expected_hit_count"]
            for trace in traces["candidate"]
        ],
        "candidate_per_run_wan_hybrid_hit_count": [
            trace.wan_hybrid_hit_count for trace in traces["candidate"]
        ],
        "reference_per_run_request_id": [
            trace.request_id for trace in traces["reference"]
        ],
        "candidate_per_run_request_id": [
            trace.request_id for trace in traces["candidate"]
        ],
        "reference_per_run_wan_hybrid_coverage": [
            trace.wan_hybrid_coverage for trace in traces["reference"]
        ],
        "candidate_per_run_wan_hybrid_coverage": [
            trace.wan_hybrid_coverage for trace in traces["candidate"]
        ],
        **result,
    }


def run_wan_transformer_forward_performance_qualification(
    *,
    model: Any,
    capture_manifest_path: str | Path,
    warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
) -> dict[str, Any]:
    """Time a real transformer_2@t521 forward in both orders without capture."""

    validate_qualification_protocol(
        comparison_mode="performance",
        run_order="both",
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    )
    fixed_input_cpu, capture_manifest = load_wan_transformer_input_capture(
        capture_manifest_path
    )
    component = capture_manifest.get("component")
    coordinates = capture_manifest.get("capture")
    if not isinstance(component, dict) or not isinstance(coordinates, dict):
        raise ValueError("Wan capture component or coordinates are missing")
    component_name = component.get("name")
    component_model_path = component.get("resolved_path")
    step_index = coordinates.get("step_index")
    actual_timestep = coordinates.get("actual_timestep")
    cfg_branch_index = coordinates.get("cfg_branch_index")
    if component_name != "transformer_2" or actual_timestep != (
        WAN_HYBRID_PROMOTION_MAX_TIMESTEP
    ):
        raise ValueError(
            "performance requires the production transformer_2@t521 capture"
        )
    if not isinstance(component_model_path, str) or not component_model_path:
        raise ValueError("captured component path is missing")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (step_index, actual_timestep, cfg_branch_index)
    ):
        raise ValueError("captured step, timestep, and CFG branch must be integers")
    _validate_loaded_model_against_capture(
        model, capture_manifest.get("model"), variant="shared"
    )
    model_device = _model_device(model)
    if model_device.type != "cuda":
        raise ValueError("performance requires the shared model on CUDA")
    configured_layers = getattr(model, "wan_hybrid_layer_indices", None)
    configured_layers = (
        sorted(configured_layers) if configured_layers is not None else None
    )
    if configured_layers != list(WAN_HYBRID_PROMOTION_LAYER_INDICES):
        raise ValueError("performance candidate must configure Wan promotion layers 37..39")
    if getattr(model, "wan_hybrid_max_timestep", None) != (
        WAN_HYBRID_PROMOTION_MAX_TIMESTEP
    ):
        raise ValueError("performance candidate must configure max_timestep=521")
    eligible_layers = _eligible_wan_hybrid_layer_indices(
        model,
        configured_layer_indices=configured_layers,
        actual_timestep=actual_timestep,
    )
    if eligible_layers != list(WAN_HYBRID_PROMOTION_LAYER_INDICES):
        raise ValueError("performance candidate promotion route is not eligible at t521")

    fixed_input_device = _move_fixed_input_to_model(
        fixed_input_cpu, device=model_device
    )
    evidence_binding = build_wan_transformer_evidence_binding(
        component_name=component_name,
        component_model_path=component_model_path,
        fixed_input=fixed_input_cpu,
        reference_model=model,
        candidate_model=model,
    )
    evidence_binding = evidence_binding | {
        "capture_manifest_sha256": capture_manifest["manifest_sha256"],
        "capture_input_sha256": capture_manifest["input_sha256"],
        "capture_request_id": capture_manifest["request_id"],
        "capture_coordinates": dict(coordinates),
        "capture_sampling_sha256": capture_manifest["sampling"]["sampling_sha256"],
    }
    evidence_binding["binding_sha256"] = _sha256_json(
        {
            key: value
            for key, value in evidence_binding.items()
            if key != "binding_sha256"
        }
    )
    stream_handle = int(torch.cuda.current_stream(model_device).cuda_stream)
    model_object_id = id(model)
    switchable_layers = _prepare_shared_performance_model(model)

    def run_variant(variant: str) -> list[WanTransformerForwardTiming]:
        _select_shared_performance_variant(switchable_layers, variant)
        return _run_timed_variant(
            model,
            fixed_input=fixed_input_device,
            variant=variant,
            component_name=component_name,
            step_index=step_index,
            actual_timestep=actual_timestep,
            cfg_branch_index=cfg_branch_index,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            device=model_device,
        )

    def summarize_variant(
        timings: Sequence[WanTransformerForwardTiming], *, variant: str
    ) -> dict[str, Any]:
        expected_layers = eligible_layers if variant == "candidate" else []
        coverage_failures = [
            failure
            for timing in timings
            if (
                failure := _direct_coverage_failure(
                    timing,
                    variant=variant,
                    candidate_layer_indices=eligible_layers,
                    num_blocks=timing.num_blocks,
                )
            )
            is not None
        ]
        summary = {
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "per_run_duration_ms": [timing.duration_ms for timing in timings],
            "median_duration_ms": statistics.median(
                timing.duration_ms for timing in timings
            ),
            "per_run_request_id": [timing.request_id for timing in timings],
            "per_run_controller_pid": [timing.controller_pid for timing in timings],
            "per_run_cuda_stream_handle": [
                timing.cuda_stream_handle for timing in timings
            ],
            "per_run_model_object_id": [model_object_id] * len(timings),
            "per_run_wan_hybrid_expected_hit_count": [len(expected_layers)]
            * len(timings),
            "per_run_wan_hybrid_hit_count": [
                timing.wan_hybrid_hit_count for timing in timings
            ],
            "per_run_wan_hybrid_coverage": [
                timing.wan_hybrid_coverage for timing in timings
            ],
            "per_run_output_summary": [timing.output_summary for timing in timings],
            "coverage_failures": coverage_failures,
        }
        if variant == "reference":
            summary["per_run_attention_backend_identity"] = [
                timing.attention_backend_identity for timing in timings
            ]
        return summary

    order_results: dict[str, Any] = {}
    reference_order_identities = []
    failures = []
    for run_order in QUALIFICATION_RUN_ORDERS:
        execution_order = (
            ("reference", "candidate")
            if run_order == "reference-first"
            else ("candidate", "reference")
        )
        timings = {}
        reference_identity = None
        for variant in execution_order:
            timings[variant] = run_variant(variant)
            if variant == "reference":
                reference_identity = _collapse_reference_attention_backend_identities(
                    [timing.attention_backend_identity for timing in timings[variant]],
                    location=f"{run_order} direct reference measured forwards",
                )
        assert reference_identity is not None
        reference_order_identities.append(reference_identity)
        reference_summary = summarize_variant(timings["reference"], variant="reference")
        candidate_summary = summarize_variant(timings["candidate"], variant="candidate")
        speedup = (
            reference_summary["median_duration_ms"]
            / candidate_summary["median_duration_ms"]
        )
        order_results[run_order] = {
            "execution_order": list(execution_order),
            "reference_attention_backend_identity": reference_identity,
            "reference_forward": reference_summary,
            "candidate_forward": candidate_summary,
            "performance": {
                "reference_median_duration_ms": reference_summary["median_duration_ms"],
                "candidate_median_duration_ms": candidate_summary["median_duration_ms"],
                "median_speedup": speedup,
            },
        }
        if speedup < MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]:
            failures.append(
                {
                    "run_order": run_order,
                    "reason": "median_speedup_below_minimum",
                    "median_speedup": speedup,
                }
            )
        for variant, summary in (
            ("reference", reference_summary),
            ("candidate", candidate_summary),
        ):
            if summary["coverage_failures"]:
                failures.append(
                    {
                        "run_order": run_order,
                        "variant": variant,
                        "reason": "request_local_coverage_failure",
                        "failures": summary["coverage_failures"],
                    }
                )
            if (
                any(
                    not math.isfinite(duration) or duration <= 0
                    for duration in summary["per_run_duration_ms"]
                )
                or any(
                    output.get("finite") is not True
                    for output in summary["per_run_output_summary"]
                )
                or summary["per_run_controller_pid"] != [os.getpid()] * measure_runs
                or summary["per_run_cuda_stream_handle"]
                != [stream_handle] * measure_runs
                or summary["per_run_model_object_id"]
                != [model_object_id] * measure_runs
            ):
                failures.append(
                    {
                        "run_order": run_order,
                        "variant": variant,
                        "reason": "invalid_duration_or_nonfinite_output",
                    }
                )

    reference_attention_backend_identity = (
        _collapse_reference_attention_backend_identities(
            reference_order_identities,
            location="direct reference performance run orders",
        )
    )
    invocation_input_sha256 = _sha256_json(_summarize_fixed_input(fixed_input_cpu))
    if evidence_binding["fixed_input_sha256"] != invocation_input_sha256:
        raise RuntimeError("full-transformer performance mutated the fixed input")
    return {
        "comparison_mode": "performance",
        "run_order": "both",
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "trajectory_capture": False,
        "correctness_evidence_scope": "separate direct correctness reports required",
        "timing_scope": (
            "synchronized complete Wan transformer forward with output materialized"
        ),
        "timing_method": (
            "time.perf_counter wall clock around a CUDA-synchronized model call; "
            "not bench_gpu_time kernel latency"
        ),
        "component_name": component_name,
        "capture_manifest_path": str(Path(capture_manifest_path).resolve()),
        "evidence_binding": evidence_binding,
        "invocation_input_sha256": invocation_input_sha256,
        "candidate_wan_hybrid_layer_indices": configured_layers,
        "candidate_wan_hybrid_eligible_layer_indices": eligible_layers,
        "candidate_wan_hybrid_max_timestep": getattr(
            model, "wan_hybrid_max_timestep", None
        ),
        "candidate_backend_exercised": True,
        "reference_attention_backend_identity": reference_attention_backend_identity,
        "execution_topology": {
            "controller_pid": os.getpid(),
            "same_python_process": True,
            "reference_model_reused_across_orders": True,
            "candidate_model_reused_across_orders": True,
            "shared_model_instance": True,
            "shared_model_reused_across_variants_and_orders": True,
            "model_object_id": model_object_id,
            "same_cuda_device": True,
            "same_fixed_input_object": True,
            "cuda_device": str(model_device),
            "same_cuda_stream_proven": True,
            "cuda_stream_handle": stream_handle,
        },
        "order_results": order_results,
        "qualification": {
            "passed": not failures,
            "thresholds": {
                "required_run_orders": list(QUALIFICATION_RUN_ORDERS),
                "warmup_runs_equals": MIN_QUALIFICATION_WARMUP_RUNS,
                "measure_runs_equals": MIN_QUALIFICATION_MEASURE_RUNS,
                "candidate_hit_count_equals": len(WAN_HYBRID_PROMOTION_LAYER_INDICES),
                "speedup_min": MODEL_QUALIFICATION_THRESHOLDS["speedup_min"],
            },
            "failures": failures,
        },
    }


def validate_wan_transformer_forward_performance_report(
    report: Any,
) -> list[str]:
    """Fail closed on the independent trajectory-off direct timing report."""

    if not isinstance(report, dict):
        return ["performance report is not a JSON object"]
    errors = []
    reference_attention_backend_identity = report.get(
        "reference_attention_backend_identity"
    )
    errors.extend(
        f"reference_attention_backend_identity: {error}"
        for error in validate_reference_attention_backend_identity(
            reference_attention_backend_identity
        )
    )
    if isinstance(reference_attention_backend_identity, dict) and (
        reference_attention_backend_identity.get("expected_instance_count") != 80
        or reference_attention_backend_identity.get("observed_instance_count") != 80
    ):
        errors.append(
            "reference_attention_backend_identity: direct Wan transformer "
            "performance requires all 80 attention instances"
        )
    expected_layers = list(WAN_HYBRID_PROMOTION_LAYER_INDICES)
    expected_hits = len(expected_layers)
    expected_scalars = {
        "comparison_mode": "performance",
        "run_order": "both",
        "warmup_runs": MIN_QUALIFICATION_WARMUP_RUNS,
        "measure_runs": MIN_QUALIFICATION_MEASURE_RUNS,
        "trajectory_capture": False,
        "correctness_evidence_scope": "separate direct correctness reports required",
        "component_name": "transformer_2",
        "candidate_wan_hybrid_layer_indices": expected_layers,
        "candidate_wan_hybrid_eligible_layer_indices": expected_layers,
        "candidate_wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
        "candidate_backend_exercised": True,
    }
    if any(report.get(name) != value for name, value in expected_scalars.items()):
        errors.append(
            "performance report does not match promotion-route transformer_2@t521"
        )
    if report.get("timing_scope") != (
        "synchronized complete Wan transformer forward with output materialized"
    ):
        errors.append("performance timing scope is incomplete")
    if report.get("timing_method") != (
        "time.perf_counter wall clock around a CUDA-synchronized model call; "
        "not bench_gpu_time kernel latency"
    ):
        errors.append("performance timing method is missing or mislabelled")
    binding = report.get("evidence_binding")
    coordinates = (
        binding.get("capture_coordinates") if isinstance(binding, dict) else None
    )
    if not isinstance(binding, dict) or binding.get("binding_sha256") != _sha256_json(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    ):
        errors.append("performance evidence binding is invalid")
    elif (
        binding.get("schema_version") != 1
        or binding.get("component_name") != "transformer_2"
        or binding.get("fixed_input_sha256") != _sha256_json(binding.get("fixed_input"))
        or report.get("invocation_input_sha256") != binding.get("fixed_input_sha256")
        or not isinstance(binding.get("capture_manifest_sha256"), str)
        or len(binding["capture_manifest_sha256"]) != 64
        or not isinstance(binding.get("capture_input_sha256"), str)
        or len(binding["capture_input_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in binding["capture_input_sha256"]
        )
        or not isinstance(binding.get("capture_request_id"), str)
        or not binding["capture_request_id"]
        or not isinstance(binding.get("capture_sampling_sha256"), str)
        or len(binding["capture_sampling_sha256"]) != 64
    ):
        errors.append("performance real-input provenance is incomplete")
    elif any(
        not isinstance(binding.get(variant), dict)
        or binding[variant].get("num_blocks") != 40
        for variant in ("reference_model", "candidate_model")
    ):
        errors.append("performance model identity does not contain all 40 blocks")
    elif binding.get("reference_model") != binding.get("candidate_model"):
        errors.append("performance variants do not bind one shared model identity")
    valid_coordinates = isinstance(coordinates, dict) and all(
        not isinstance(coordinates.get(name), bool)
        and isinstance(coordinates.get(name), int)
        for name in ("step_index", "actual_timestep", "cfg_branch_index")
    )
    if not valid_coordinates or coordinates.get("actual_timestep") != (
        WAN_HYBRID_PROMOTION_MAX_TIMESTEP
    ):
        errors.append("performance capture is not the real t521 input")
    topology = report.get("execution_topology")
    if not isinstance(topology, dict) or any(
        topology.get(name) is not True
        for name in (
            "same_python_process",
            "reference_model_reused_across_orders",
            "candidate_model_reused_across_orders",
            "shared_model_instance",
            "shared_model_reused_across_variants_and_orders",
            "same_cuda_device",
            "same_fixed_input_object",
            "same_cuda_stream_proven",
        )
    ):
        errors.append("performance process/stream topology is not proven")
    elif (
        isinstance(topology.get("controller_pid"), bool)
        or not isinstance(topology.get("controller_pid"), int)
        or isinstance(topology.get("cuda_stream_handle"), bool)
        or not isinstance(topology.get("cuda_stream_handle"), int)
        or isinstance(topology.get("model_object_id"), bool)
        or not isinstance(topology.get("model_object_id"), int)
    ):
        errors.append("performance process/stream identity is missing")
    elif (
        topology.get("direct_in_process_models") != 1
        or topology.get("scheduler_worker_processes") != 0
        or topology.get("scheduler_ports_reserved_for_variant_configuration_only")
        is not True
    ):
        errors.append("performance direct-process topology is incomplete")

    order_results = report.get("order_results")
    if not isinstance(order_results, dict):
        errors.append("performance dual-order results are missing")
    else:
        for run_order in QUALIFICATION_RUN_ORDERS:
            order = order_results.get(run_order)
            if not isinstance(order, dict):
                errors.append(f"{run_order}: performance result is missing")
                continue
            expected_order = (
                ["reference", "candidate"]
                if run_order == "reference-first"
                else ["candidate", "reference"]
            )
            if order.get("execution_order") != expected_order:
                errors.append(f"{run_order}: execution order is invalid")
            order_reference_identity = order.get(
                "reference_attention_backend_identity"
            )
            if order_reference_identity != reference_attention_backend_identity:
                errors.append(
                    f"{run_order}: reference attention backend identity is inconsistent"
                )
            errors.extend(
                f"{run_order}.reference_attention_backend_identity: {error}"
                for error in validate_reference_attention_backend_identity(
                    order_reference_identity
                )
            )
            for variant in ("reference", "candidate"):
                summary = order.get(f"{variant}_forward")
                if not isinstance(summary, dict):
                    errors.append(f"{run_order}.{variant}: forward summary is missing")
                    continue
                durations = summary.get("per_run_duration_ms")
                request_ids = summary.get("per_run_request_id")
                output_summaries = summary.get("per_run_output_summary")
                coverages = summary.get("per_run_wan_hybrid_coverage")
                per_run_attention_backend_identities = summary.get(
                    "per_run_attention_backend_identity"
                )
                expected_variant_hits = expected_hits if variant == "candidate" else 0
                expected_pid = (
                    topology.get("controller_pid")
                    if isinstance(topology, dict)
                    else None
                )
                expected_stream = (
                    topology.get("cuda_stream_handle")
                    if isinstance(topology, dict)
                    else None
                )
                expected_model_object_id = (
                    topology.get("model_object_id")
                    if isinstance(topology, dict)
                    else None
                )
                if (
                    summary.get("warmup_runs") != MIN_QUALIFICATION_WARMUP_RUNS
                    or summary.get("measure_runs") != MIN_QUALIFICATION_MEASURE_RUNS
                    or not isinstance(durations, list)
                    or len(durations) != MIN_QUALIFICATION_MEASURE_RUNS
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        for value in durations
                    )
                    or not isinstance(request_ids, list)
                    or len(request_ids) != MIN_QUALIFICATION_MEASURE_RUNS
                    or len(set(request_ids)) != MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("per_run_controller_pid")
                    != [expected_pid] * MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("per_run_cuda_stream_handle")
                    != [expected_stream] * MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("per_run_model_object_id")
                    != [expected_model_object_id] * MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("per_run_wan_hybrid_expected_hit_count")
                    != [expected_variant_hits] * MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("per_run_wan_hybrid_hit_count")
                    != [expected_variant_hits] * MIN_QUALIFICATION_MEASURE_RUNS
                    or summary.get("coverage_failures") != []
                    or not isinstance(coverages, list)
                    or len(coverages) != MIN_QUALIFICATION_MEASURE_RUNS
                    or not isinstance(output_summaries, list)
                    or len(output_summaries) != MIN_QUALIFICATION_MEASURE_RUNS
                    or any(
                        not isinstance(output, dict) or output.get("finite") is not True
                        for output in output_summaries or []
                    )
                    or (
                        variant == "reference"
                        and per_run_attention_backend_identities
                        != [reference_attention_backend_identity]
                        * MIN_QUALIFICATION_MEASURE_RUNS
                    )
                    or (
                        variant == "candidate"
                        and "per_run_attention_backend_identity" in summary
                    )
                ):
                    errors.append(f"{run_order}.{variant}: forward evidence is invalid")
                elif valid_coordinates:
                    if summary.get("median_duration_ms") != statistics.median(
                        durations
                    ):
                        errors.append(
                            f"{run_order}.{variant}: median duration is inconsistent"
                        )
                    for run_index, (request_id, coverage, output_summary) in enumerate(
                        zip(request_ids, coverages, output_summaries)
                    ):
                        timing = WanTransformerForwardTiming(
                            duration_ms=durations[run_index],
                            request_id=request_id,
                            component_name="transformer_2",
                            step_index=coordinates["step_index"],
                            actual_timestep=coordinates["actual_timestep"],
                            cfg_branch_index=coordinates["cfg_branch_index"],
                            num_blocks=40,
                            controller_pid=expected_pid,
                            cuda_stream_handle=expected_stream,
                            wan_hybrid_coverage=coverage,
                            output_summary=output_summary,
                        )
                        failure = _direct_coverage_failure(
                            timing,
                            variant=variant,
                            candidate_layer_indices=expected_layers,
                            num_blocks=40,
                        )
                        if failure is not None:
                            errors.append(
                                f"{run_order}.{variant}[{run_index}]: serialized "
                                "coverage is invalid"
                            )
            speedup = order.get("performance", {}).get("median_speedup")
            if (
                isinstance(speedup, bool)
                or not isinstance(speedup, (int, float))
                or not math.isfinite(speedup)
                or speedup < MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]
            ):
                errors.append(f"{run_order}: median speedup is below 1.0")
            else:
                reference_summary = order.get("reference_forward")
                candidate_summary = order.get("candidate_forward")
                if isinstance(reference_summary, dict) and isinstance(
                    candidate_summary, dict
                ):
                    reference_median = reference_summary.get("median_duration_ms")
                    candidate_median = candidate_summary.get("median_duration_ms")
                    performance = order.get("performance")
                    if (
                        not isinstance(performance, dict)
                        or performance.get("reference_median_duration_ms")
                        != reference_median
                        or performance.get("candidate_median_duration_ms")
                        != candidate_median
                        or isinstance(reference_median, bool)
                        or not isinstance(reference_median, (int, float))
                        or isinstance(candidate_median, bool)
                        or not isinstance(candidate_median, (int, float))
                        or candidate_median <= 0
                        or not math.isclose(
                            speedup,
                            reference_median / candidate_median,
                            rel_tol=1e-12,
                            abs_tol=0.0,
                        )
                    ):
                        errors.append(f"{run_order}: timing comparison is inconsistent")
    qualification = report.get("qualification")
    expected_thresholds = {
        "required_run_orders": list(QUALIFICATION_RUN_ORDERS),
        "warmup_runs_equals": MIN_QUALIFICATION_WARMUP_RUNS,
        "measure_runs_equals": MIN_QUALIFICATION_MEASURE_RUNS,
        "candidate_hit_count_equals": expected_hits,
        "speedup_min": MODEL_QUALIFICATION_THRESHOLDS["speedup_min"],
    }
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("failures") != []
        or qualification.get("thresholds") != expected_thresholds
    ):
        errors.append("performance qualification is incomplete")
    if "cross_variant_metrics" in report or "repeatability" in report:
        errors.append("performance report contains correctness trajectory data")
    port_provenance = report.get("port_provenance")
    if not isinstance(port_provenance, dict):
        errors.append("performance port provenance is missing")
    else:
        ports = [
            port_provenance.get("master_port"),
            port_provenance.get("reference_scheduler_port"),
            port_provenance.get("candidate_scheduler_port"),
        ]
        if (
            any(
                isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= 65535
                for port in ports
            )
            or len(set(ports)) != 3
            or port_provenance.get("reference_strict_ports") is not True
            or port_provenance.get("candidate_strict_ports") is not True
        ):
            errors.append("performance ports are not explicit, strict, and distinct")
        if isinstance(topology, dict) and topology.get("port_topology") != (
            port_provenance
        ):
            errors.append("performance topology is not bound to port provenance")
    return errors


def validate_wan_transformer_forward_report(
    report: Any,
    *,
    expected_warmup_runs: int = MIN_QUALIFICATION_WARMUP_RUNS,
    expected_measure_runs: int = MIN_QUALIFICATION_MEASURE_RUNS,
    expected_model_path: str | Path | None = None,
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
    if report.get("component_name") not in ("transformer", "transformer_2"):
        errors.append("component_name must identify transformer or transformer_2")
    binding = report.get("evidence_binding")
    expected_capture_coordinates: dict[str, int] | None = None
    if not isinstance(binding, dict):
        errors.append("evidence binding is missing")
    else:
        binding_payload = {
            key: value for key, value in binding.items() if key != "binding_sha256"
        }
        if binding.get("binding_sha256") != _sha256_json(binding_payload):
            errors.append("evidence binding SHA256 is inconsistent")
        if binding.get("schema_version") != 1:
            errors.append("evidence binding schema is unsupported")
        if binding.get("component_name") != report.get("component_name"):
            errors.append("evidence binding component does not match report")
        fixed_input = binding.get("fixed_input")
        if fixed_input is None or binding.get("fixed_input_sha256") != _sha256_json(
            fixed_input
        ):
            errors.append("fixed-input provenance is incomplete")
        if report.get("invocation_input_sha256") != binding.get("fixed_input_sha256"):
            errors.append("invoked input does not match fixed-input provenance")
        capture_input_sha256 = binding.get("capture_input_sha256")
        if (
            not isinstance(capture_input_sha256, str)
            or len(capture_input_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in capture_input_sha256
            )
        ):
            errors.append("capture input provenance is incomplete")
        capture_coordinates = binding.get("capture_coordinates")
        if not (
            isinstance(binding.get("capture_manifest_sha256"), str)
            and len(binding["capture_manifest_sha256"]) == 64
            and isinstance(binding.get("capture_request_id"), str)
            and binding["capture_request_id"]
            and isinstance(binding.get("capture_sampling_sha256"), str)
            and len(binding["capture_sampling_sha256"]) == 64
            and isinstance(capture_coordinates, dict)
            and all(
                not isinstance(capture_coordinates.get(name), bool)
                and isinstance(capture_coordinates.get(name), int)
                for name in (
                    "step_index",
                    "actual_timestep",
                    "cfg_branch_index",
                )
            )
        ):
            errors.append("worker capture binding is incomplete")
        else:
            expected_capture_coordinates = {
                name: capture_coordinates[name]
                for name in (
                    "step_index",
                    "actual_timestep",
                    "cfg_branch_index",
                )
            }
        for variant in ("reference_model", "candidate_model"):
            identity = binding.get(variant)
            if not isinstance(identity, dict):
                errors.append(f"{variant} identity is missing")
                continue
            identity_payload = {
                key: value
                for key, value in identity.items()
                if key != "identity_sha256"
            }
            if identity.get("identity_sha256") != _sha256_json(identity_payload):
                errors.append(f"{variant} identity SHA256 is inconsistent")
            if identity.get("num_blocks") != 40:
                errors.append(f"{variant} identity does not contain 40 blocks")
        config_files = binding.get("component_config_files")
        if not isinstance(config_files, list) or not any(
            isinstance(item, dict)
            and item.get("path") == "config.json"
            and isinstance(item.get("sha256"), str)
            and len(item["sha256"]) == 64
            for item in config_files
        ):
            errors.append("component config provenance is incomplete")
        if expected_model_path is not None and report.get("component_name") in (
            "transformer",
            "transformer_2",
        ):
            expected_component_path = (
                Path(expected_model_path)
                .expanduser()
                .joinpath(report["component_name"])
                .resolve()
            )
            if binding.get("resolved_component_path") != str(expected_component_path):
                errors.append("resolved component path does not match model/component")
            expected_config_path = expected_component_path / "config.json"
            reported_config = next(
                (
                    item
                    for item in config_files
                    if isinstance(item, dict) and item.get("path") == "config.json"
                ),
                None,
            )
            if (
                not expected_config_path.is_file()
                or reported_config is None
                or reported_config.get("sha256") != _sha256_file(expected_config_path)
            ):
                errors.append(
                    "component config provenance does not match resolved model"
                )
    num_blocks = report.get("num_blocks")
    if (
        isinstance(num_blocks, bool)
        or not isinstance(num_blocks, int)
        or num_blocks != 40
    ):
        errors.append("num_blocks must equal the Wan serving depth of 40")
    candidate_layer_indices = report.get("candidate_wan_hybrid_layer_indices")
    if (
        not isinstance(candidate_layer_indices, list)
        or not candidate_layer_indices
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in candidate_layer_indices
        )
        or candidate_layer_indices != sorted(set(candidate_layer_indices))
        or any(not 0 <= index < 40 for index in candidate_layer_indices)
    ):
        errors.append("candidate Wan hybrid layer indices are invalid")
        candidate_layer_indices = []
    candidate_eligible_layer_indices = report.get(
        "candidate_wan_hybrid_eligible_layer_indices"
    )
    if (
        not isinstance(candidate_eligible_layer_indices, list)
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in candidate_eligible_layer_indices
        )
        or candidate_eligible_layer_indices
        != sorted(set(candidate_eligible_layer_indices))
        or any(
            index not in candidate_layer_indices
            for index in candidate_eligible_layer_indices
        )
    ):
        errors.append("candidate Wan hybrid eligible layer indices are invalid")
        candidate_eligible_layer_indices = []
    candidate_expected_hits = len(candidate_eligible_layer_indices)
    candidate_backend_exercised = candidate_expected_hits > 0
    if report.get("candidate_backend_exercised") is not candidate_backend_exercised:
        errors.append("candidate backend exercised status is invalid")
    candidate_backend_expectation = report.get("candidate_backend_expectation")
    if candidate_backend_expectation not in ("exercised", "temporal_fallback"):
        errors.append("candidate backend expectation is invalid")
    elif candidate_backend_expectation == "exercised":
        if not candidate_backend_exercised:
            errors.append("exercised candidate expectation requires eligible layers")
    else:
        max_timestep = report.get("candidate_wan_hybrid_max_timestep")
        actual_timestep = (
            expected_capture_coordinates.get("actual_timestep")
            if isinstance(expected_capture_coordinates, dict)
            else None
        )
        if candidate_backend_exercised:
            errors.append("temporal fallback expectation forbids eligible layers")
        if (
            isinstance(max_timestep, bool)
            or not isinstance(max_timestep, (int, float))
            or isinstance(actual_timestep, bool)
            or not isinstance(actual_timestep, int)
            or actual_timestep <= max_timestep
        ):
            errors.append(
                "temporal fallback expectation requires actual timestep above max"
            )

    expected_cross_pairs = set(
        itertools.product(range(expected_measure_runs), range(expected_measure_runs))
    )
    expected_repeat_pairs = set(itertools.combinations(range(expected_measure_runs), 2))

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
                trajectory.get("num_steps") if isinstance(trajectory, dict) else None
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
    expected_hit_counts = report.get("candidate_per_run_wan_hybrid_expected_hit_count")
    if (
        not isinstance(hit_counts, list)
        or len(hit_counts) != expected_measure_runs
        or not isinstance(expected_hit_counts, list)
        or len(expected_hit_counts) != expected_measure_runs
        or any(hit_count != candidate_expected_hits for hit_count in hit_counts)
        or any(
            expected_hit_count != candidate_expected_hits
            for expected_hit_count in expected_hit_counts
        )
    ):
        errors.append(
            "every measured candidate hit count must equal the eligible depth"
        )

    request_ids_by_variant = {}
    coverages_by_variant = {}
    for variant in ("reference", "candidate"):
        request_ids = report.get(f"{variant}_per_run_request_id")
        coverages = report.get(f"{variant}_per_run_wan_hybrid_coverage")
        if (
            not isinstance(request_ids, list)
            or len(request_ids) != expected_measure_runs
            or any(not isinstance(item, str) or not item for item in request_ids)
            or len(set(request_ids)) != expected_measure_runs
        ):
            errors.append(f"{variant} measured request IDs are not unique")
            request_ids = []
        if not isinstance(coverages, list) or len(coverages) != expected_measure_runs:
            errors.append(f"{variant} request-local coverage is incomplete")
            coverages = []
        request_ids_by_variant[variant] = request_ids
        coverages_by_variant[variant] = coverages
    if set(request_ids_by_variant.get("reference", ())) & set(
        request_ids_by_variant.get("candidate", ())
    ):
        errors.append("reference and candidate request IDs overlap")
    for variant, coverages in coverages_by_variant.items():
        candidate = variant == "candidate"
        expected_hybrid_layers = candidate_eligible_layer_indices if candidate else []
        expected_hits = len(expected_hybrid_layers)
        for run_index, coverage in enumerate(coverages):
            location = f"{variant}_per_run_wan_hybrid_coverage[{run_index}]"
            if not isinstance(coverage, dict):
                errors.append(f"{location}: coverage is not an object")
                continue
            expected_request_ids = request_ids_by_variant.get(variant, [])
            expected_request_id = (
                expected_request_ids[run_index]
                if run_index < len(expected_request_ids)
                else None
            )
            scalars = {
                "schema_version": 2,
                "request_id": expected_request_id,
                "expected_hit_count": expected_hits,
                "actual_hit_count": expected_hits,
                "attributed_actual_hit_count": expected_hits,
                "unattributed_actual_hit_count": 0,
                "eligible_hybrid_miss_count": 0,
                "num_route_events": 40,
                "num_success_events": expected_hits,
            }
            if any(coverage.get(name) != value for name, value in scalars.items()):
                errors.append(f"{location}: request-local scalar coverage is invalid")
                continue
            boundary_errors = validate_wan_hybrid_exact_serving_boundary_evidence(
                coverage
            )
            if boundary_errors:
                errors.extend(
                    f"{location}: {error}" for error in boundary_errors
                )
                continue
            steps = coverage.get("steps")
            step = (
                steps[0]
                if isinstance(steps, list)
                and len(steps) == 1
                and isinstance(steps[0], dict)
                else None
            )
            if step is None or expected_capture_coordinates is None:
                errors.append(f"{location}: step/branch coverage is invalid")
                continue
            expected_step = {
                "step_index": expected_capture_coordinates["step_index"],
                "actual_timestep": expected_capture_coordinates["actual_timestep"],
                "active_component": report.get("component_name"),
                "executed_cfg_branch_indices": [
                    expected_capture_coordinates["cfg_branch_index"]
                ],
            }
            if any(step.get(name) != value for name, value in expected_step.items()):
                errors.append(f"{location}: capture coordinates are invalid")
                continue
            branches = step.get("branches")
            if not isinstance(branches, list) or len(branches) != 1:
                errors.append(f"{location}: step/branch coverage is invalid")
                continue
            branch = branches[0]
            expected_layers = list(range(40))
            expected_routes = {
                "cfg_branch_index": expected_capture_coordinates["cfg_branch_index"],
                "num_layers": 40,
                "layer_indices": expected_layers,
                "eligible_layer_indices": expected_hybrid_layers,
                "planned_hybrid_layer_indices": expected_hybrid_layers,
                "successful_hybrid_layer_indices": expected_hybrid_layers,
                "eligible_hybrid_miss_layer_indices": [],
                "unexpected_successful_hybrid_layer_indices": [],
                "configured_fallback_layer_indices": (
                    [
                        index
                        for index in expected_layers
                        if index not in expected_hybrid_layers
                    ]
                    if candidate
                    else []
                ),
                "control_layer_indices": [] if candidate else expected_layers,
                "expected_hit_count": expected_hits,
                "actual_hit_count": expected_hits,
            }
            if any(
                branch.get(name) != value for name, value in expected_routes.items()
            ):
                errors.append(f"{location}: planned/success route coverage is invalid")

    qualification = report.get("qualification")
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("failures") != []
        or qualification.get("thresholds") != MODEL_QUALIFICATION_THRESHOLDS
    ):
        errors.append("correctness qualification is incomplete")
    elif qualification.get("candidate_backend_hits") != (
        {
            "passed": True,
            "thresholds": {
                "candidate_hit_count_equals_expected": True,
                "expected_hit_count_min_exclusive": 0,
            },
            "expected_hit_counts": [candidate_expected_hits] * expected_measure_runs,
            "actual_hit_counts": [candidate_expected_hits] * expected_measure_runs,
            "failures": [],
            "candidate_backend_exercised": True,
        }
        if candidate_backend_expectation == "exercised"
        else {
            "passed": True,
            "thresholds": {
                "candidate_hit_count_equals_expected": True,
                "expected_hit_count_equals": 0,
            },
            "expected_hit_counts": [0] * expected_measure_runs,
            "actual_hit_counts": [0] * expected_measure_runs,
            "failures": [],
            "candidate_backend_exercised": False,
        }
    ):
        errors.append("candidate backend-hit qualification is incomplete")
    elif qualification.get("request_local_backend_coverage") != {
        "passed": True,
        "failures": [],
    }:
        errors.append("request-local backend coverage qualification is incomplete")
    return errors


def write_wan_transformer_forward_report(
    report: dict[str, Any], output_path: str | Path
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
