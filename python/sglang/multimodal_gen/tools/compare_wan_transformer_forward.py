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
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    WanHybridEvidenceCollector,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    _model_identity,
    load_wan_transformer_input_capture,
)
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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
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
    config_files = [
        {"path": "config.json", "sha256": _sha256_file(config_path)}
    ]
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


def _direct_coverage_failure(
    trace: WanTransformerForwardTrace,
    *,
    variant: str,
) -> dict[str, Any] | None:
    coverage = trace.wan_hybrid_coverage
    expected_layers = list(range(len(trace.block_outputs)))
    candidate = variant == "candidate"
    expected_hits = len(expected_layers) if candidate else 0
    expected_successes = expected_hits
    if coverage.get("request_id") != trace.request_id:
        return {"variant": variant, "reason": "request_id_mismatch"}
    expected_scalars = {
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
            "actual": {
                name: coverage.get(name) for name in expected_scalars
            },
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
        "eligible_layer_indices": expected_layers if candidate else [],
        "planned_hybrid_layer_indices": expected_layers if candidate else [],
        "successful_hybrid_layer_indices": expected_layers if candidate else [],
        "eligible_hybrid_miss_layer_indices": [],
        "unexpected_successful_hybrid_layer_indices": [],
        "configured_fallback_layer_indices": [],
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
        [len(trace.block_outputs) for trace in candidate_traces],
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
    coverage_failures = [
        failure
        for variant, variant_traces in (
            ("reference", reference_traces),
            ("candidate", candidate_traces),
        )
        for trace in variant_traces
        if (failure := _direct_coverage_failure(trace, variant=variant))
        is not None
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
        "capture_request_id": capture_manifest["request_id"],
        "capture_coordinates": dict(capture_coordinates),
        "capture_sampling_sha256": capture_manifest["sampling"][
            "sampling_sha256"
        ],
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
    invocation_input_sha256 = _sha256_json(
        _summarize_fixed_input(fixed_input_cpu)
    )
    if evidence_binding["fixed_input_sha256"] != invocation_input_sha256:
        raise RuntimeError("full-transformer forward mutated the fixed input")

    result = evaluate_transformer_forward_correctness(
        traces["reference"], traces["candidate"]
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
        if (
            fixed_input is None
            or binding.get("fixed_input_sha256") != _sha256_json(fixed_input)
        ):
            errors.append("fixed-input provenance is incomplete")
        if report.get("invocation_input_sha256") != binding.get(
            "fixed_input_sha256"
        ):
            errors.append("invoked input does not match fixed-input provenance")
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
        if (
            not isinstance(config_files, list)
            or not any(
                isinstance(item, dict)
                and item.get("path") == "config.json"
                and isinstance(item.get("sha256"), str)
                and len(item["sha256"]) == 64
                for item in config_files
            )
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
            if binding.get("resolved_component_path") != str(
                expected_component_path
            ):
                errors.append(
                    "resolved component path does not match model/component"
                )
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
                or reported_config.get("sha256")
                != _sha256_file(expected_config_path)
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
    expected_hit_counts = report.get(
        "candidate_per_run_wan_hybrid_expected_hit_count"
    )
    if (
        not isinstance(hit_counts, list)
        or len(hit_counts) != expected_measure_runs
        or not isinstance(expected_hit_counts, list)
        or len(expected_hit_counts) != expected_measure_runs
        or any(hit_count != 40 for hit_count in hit_counts)
        or any(expected_hit_count != 40 for expected_hit_count in expected_hit_counts)
    ):
        errors.append("every measured candidate hit count must equal expected depth 40")

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
        expected_hits = 40 if candidate else 0
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
                "actual_timestep": expected_capture_coordinates[
                    "actual_timestep"
                ],
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
                "cfg_branch_index": expected_capture_coordinates[
                    "cfg_branch_index"
                ],
                "num_layers": 40,
                "layer_indices": expected_layers,
                "eligible_layer_indices": expected_layers if candidate else [],
                "planned_hybrid_layer_indices": (
                    expected_layers if candidate else []
                ),
                "successful_hybrid_layer_indices": (
                    expected_layers if candidate else []
                ),
                "eligible_hybrid_miss_layer_indices": [],
                "unexpected_successful_hybrid_layer_indices": [],
                "configured_fallback_layer_indices": [],
                "control_layer_indices": [] if candidate else expected_layers,
                "expected_hit_count": expected_hits,
                "actual_hit_count": expected_hits,
            }
            if any(
                branch.get(name) != value
                for name, value in expected_routes.items()
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
    elif qualification.get("candidate_backend_hits") != {
        "passed": True,
        "thresholds": {
            "candidate_hit_count_equals_expected": True,
            "expected_hit_count_min_exclusive": 0,
        },
        "expected_hit_counts": [40] * expected_measure_runs,
        "actual_hit_counts": [40] * expected_measure_runs,
        "failures": [],
    }:
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
