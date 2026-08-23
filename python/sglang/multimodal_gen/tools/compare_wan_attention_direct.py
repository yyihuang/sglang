"""Qualify public Wan hybrid attention directly on real post-RoPE Q/K/V.

The tool loads one captured Wan component input, runs the FA reference model once
to intercept raw post-RoPE BF16 Q/K/V at selected self-attention layers, then
invokes the reference and public Wan hybrid attention modules directly.  This is
a correctness-only diagnostic; its warmups and repeated calls are not a latency
measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    read_wan_hybrid_hit_count,
    reset_wan_hybrid_hit_count,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    load_wan_transformer_input_capture,
)
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
    compute_tensor_metrics,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    WanTransformerDirectRequest,
    _model_device,
    _move_fixed_input_to_model,
    _validate_loaded_model_against_capture,
    _wan_transformer_autocast_context,
)
from sglang.multimodal_gen.tools.run_wan_transformer_forward_report import (
    _initialize_single_gpu_runtime,
    _load_component,
    build_direct_port_provenance,
)

TAIL5_LAYER_INDICES = (35, 36, 37, 38, 39)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--run-order",
        choices=("reference-first", "candidate-first"),
        required=True,
    )
    parser.add_argument("--model-id")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--reference-scheduler-port", type=int, required=True)
    parser.add_argument("--candidate-scheduler-port", type=int, required=True)
    parser.add_argument("--strict-ports", action="store_true")
    parser.add_argument("--reference-attention-backend", default="fa")
    parser.add_argument("--candidate-attention-backend", default="wan_hybrid")
    parser.add_argument("--candidate-attention-backend-config", required=True)
    parser.add_argument("--layer-index", action="append", type=int, dest="layers")
    parser.add_argument("--expected-actual-timestep", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    return parser


def _capture_post_rope_qkv(
    model: Any,
    *,
    fixed_input: dict[str, Any],
    layer_indices: Sequence[int],
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_index: int):
        def capture(_module, args):
            if len(args) < 3:
                raise RuntimeError("Wan self-attention hook did not receive Q/K/V")
            if layer_index in captured:
                raise RuntimeError(f"Wan layer {layer_index} executed more than once")
            q, k, v = args[:3]
            captured[layer_index] = tuple(
                tensor.detach().clone() for tensor in (q, k, v)
            )

        return capture

    for layer_index in layer_indices:
        hooks.append(
            model.blocks[layer_index].attn1.register_forward_pre_hook(
                make_hook(layer_index)
            )
        )
    try:
        request_id = f"wan-direct-qkv-capture-{uuid.uuid4()}"
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
                wan_hybrid_evidence_collector=None,
            ),
        ):
            model(**fixed_input)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != set(layer_indices):
        raise RuntimeError(
            f"Wan Q/K/V capture mismatch: expected {list(layer_indices)}, "
            f"captured {sorted(captured)}"
        )
    for layer_index, tensors in captured.items():
        for name, tensor in zip(("q", "k", "v"), tensors):
            if tensor.shape != (1, 4800, 40, 128):
                raise RuntimeError(
                    f"layer {layer_index} {name} has shape {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise RuntimeError(
                    f"layer {layer_index} {name} must be contiguous BF16 NHD"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"layer {layer_index} {name} is not finite")
    return captured


def _run_direct_variant(
    model: Any,
    *,
    qkv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    variant: str,
    component_name: str,
    step_index: int,
    actual_timestep: int,
    cfg_branch_index: int,
    warmup_runs: int,
    measure_runs: int,
) -> tuple[list[dict[int, torch.Tensor]], list[int], dict[int, list[int]]]:
    attention_by_layer = {
        layer_index: model.blocks[layer_index].attn1 for layer_index in qkv_by_layer
    }

    def run_once(*, retain: bool):
        reset_wan_hybrid_hit_count()
        outputs = {}
        pointers = {}
        request_id = f"wan-direct-attention-{variant}-{uuid.uuid4()}"
        with (
            torch.inference_mode(),
            set_forward_context(
                current_timestep=step_index,
                attn_metadata=None,
                forward_batch=WanTransformerDirectRequest(request_id=request_id),
                wan_component_name=component_name,
                wan_actual_timestep=actual_timestep,
                wan_cfg_branch_index=cfg_branch_index,
                wan_hybrid_evidence_collector=None,
            ),
        ):
            for layer_index, attention in attention_by_layer.items():
                output = attention(*qkv_by_layer[layer_index])
                if output.dtype != torch.bfloat16:
                    raise RuntimeError("Wan direct attention output must be BF16")
                if output.shape != (1, 4800, 40, 128):
                    raise RuntimeError("Wan direct attention output shape is invalid")
                if not torch.isfinite(output).all():
                    raise RuntimeError("Wan direct attention output is not finite")
                if variant == "candidate":
                    caller_output = getattr(attention.attn_impl, "_output", None)
                    if (
                        caller_output is None
                        or output.data_ptr() != caller_output.data_ptr()
                    ):
                        raise RuntimeError(
                            "Wan public API did not materialize into caller-owned output"
                        )
                    pointers[layer_index] = output.data_ptr()
                if retain:
                    outputs[layer_index] = output.detach().cpu().clone()
        hit_count = read_wan_hybrid_hit_count()
        expected_hits = len(qkv_by_layer) if variant == "candidate" else 0
        if hit_count != expected_hits:
            raise RuntimeError(
                f"{variant} direct hit count {hit_count} != {expected_hits}"
            )
        return outputs, hit_count, pointers

    for _ in range(warmup_runs):
        run_once(retain=False)
    measured = []
    hits = []
    pointer_history = {layer_index: [] for layer_index in qkv_by_layer}
    for _ in range(measure_runs):
        outputs, hit_count, pointers = run_once(retain=True)
        measured.append(outputs)
        hits.append(hit_count)
        for layer_index, pointer in pointers.items():
            pointer_history[layer_index].append(pointer)
    return measured, hits, pointer_history


def _envelope(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "min_cosine": min(metric["cosine_similarity"] for metric in metrics),
        "max_mae": max(metric["mae"] for metric in metrics),
        "max_abs": max(metric["max_abs"] for metric in metrics),
        "all_finite": all(metric["finite"] for metric in metrics),
        "all_within_tolerance": all(metric["within_tolerance"] for metric in metrics),
        "all_exact": all(metric["exact_match"] for metric in metrics),
    }


def _summarize_layer(
    reference_runs: Sequence[dict[int, torch.Tensor]],
    candidate_runs: Sequence[dict[int, torch.Tensor]],
    *,
    layer_index: int,
) -> dict[str, Any]:
    cross = [
        compute_tensor_metrics(reference[layer_index], candidate[layer_index])
        | {
            "reference_run_index": reference_index,
            "candidate_run_index": candidate_index,
        }
        for reference_index, reference in enumerate(reference_runs)
        for candidate_index, candidate in enumerate(candidate_runs)
    ]
    repeatability = {}
    for variant, runs in (
        ("reference", reference_runs),
        ("candidate", candidate_runs),
    ):
        comparisons = [
            compute_tensor_metrics(runs[lhs][layer_index], runs[rhs][layer_index])
            | {"lhs_run_index": lhs, "rhs_run_index": rhs}
            for lhs, rhs in itertools.combinations(range(len(runs)), 2)
        ]
        repeatability[variant] = {
            "num_pairs": len(comparisons),
            "envelope": _envelope(comparisons),
            "comparisons": comparisons,
        }
    return {
        "layer_index": layer_index,
        "cross_variant": {
            "num_pairs": len(cross),
            "envelope": _envelope(cross),
            "comparisons": cross,
        },
        "repeatability": repeatability,
    }


def main() -> None:
    args = _build_parser().parse_args()
    layers = tuple(args.layers or TAIL5_LAYER_INDICES)
    if layers != TAIL5_LAYER_INDICES:
        raise ValueError(f"direct tail5 qualification requires {TAIL5_LAYER_INDICES}")
    if args.warmup_runs != 2 or args.measure_runs != 5:
        raise ValueError("direct qualification requires warmup=2 and measure=5")
    port_provenance = build_direct_port_provenance(
        master_port=args.master_port,
        reference_scheduler_port=args.reference_scheduler_port,
        candidate_scheduler_port=args.candidate_scheduler_port,
        strict_ports=args.strict_ports,
    )
    config = json.loads(args.candidate_attention_backend_config)
    if config != {
        "wan_hybrid_max_timestep": 521,
        "wan_hybrid_layer_indices": list(TAIL5_LAYER_INDICES),
    }:
        raise ValueError("candidate configuration must be the fixed t521 tail5 route")

    capture_path = Path(args.capture_manifest).expanduser().resolve()
    fixed_input_cpu, manifest = load_wan_transformer_input_capture(capture_path)
    coordinates = manifest["capture"]
    component = manifest["component"]
    component_name = component["name"]
    actual_timestep = coordinates["actual_timestep"]
    if actual_timestep != args.expected_actual_timestep:
        raise ValueError("capture actual timestep does not match the expected timestep")

    _initialize_single_gpu_runtime(args.master_port)
    common = {
        "model_root": manifest["model_root"],
        "component_name": component_name,
        "component_path": component["resolved_path"],
        "model_id": args.model_id,
        "transformer_weights_path": None,
        "strict_ports": args.strict_ports,
    }
    reference_model = _load_component(
        **common,
        attention_backend=args.reference_attention_backend,
        attention_backend_config=None,
        scheduler_port=args.reference_scheduler_port,
    )
    candidate_model = _load_component(
        **common,
        attention_backend=args.candidate_attention_backend,
        attention_backend_config=args.candidate_attention_backend_config,
        scheduler_port=args.candidate_scheduler_port,
    )
    _validate_loaded_model_against_capture(
        reference_model, manifest["model"], variant="reference"
    )
    _validate_loaded_model_against_capture(
        candidate_model, manifest["model"], variant="candidate"
    )
    reference_input = _move_fixed_input_to_model(
        fixed_input_cpu, device=_model_device(reference_model)
    )
    qkv_by_layer = _capture_post_rope_qkv(
        reference_model,
        fixed_input=reference_input,
        layer_indices=layers,
        component_name=component_name,
        step_index=coordinates["step_index"],
        actual_timestep=actual_timestep,
        cfg_branch_index=coordinates["cfg_branch_index"],
    )

    calls = {
        "reference": lambda: _run_direct_variant(
            reference_model,
            qkv_by_layer=qkv_by_layer,
            variant="reference",
            component_name=component_name,
            step_index=coordinates["step_index"],
            actual_timestep=actual_timestep,
            cfg_branch_index=coordinates["cfg_branch_index"],
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        ),
        "candidate": lambda: _run_direct_variant(
            candidate_model,
            qkv_by_layer=qkv_by_layer,
            variant="candidate",
            component_name=component_name,
            step_index=coordinates["step_index"],
            actual_timestep=actual_timestep,
            cfg_branch_index=coordinates["cfg_branch_index"],
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        ),
    }
    order = (
        ("reference", "candidate")
        if args.run_order == "reference-first"
        else ("candidate", "reference")
    )
    results = {}
    for variant in order:
        results[variant] = calls[variant]()
    reference_runs, reference_hits, _ = results["reference"]
    candidate_runs, candidate_hits, candidate_pointers = results["candidate"]
    layer_reports = [
        _summarize_layer(reference_runs, candidate_runs, layer_index=layer_index)
        for layer_index in layers
    ]
    failures = []
    for layer in layer_reports:
        envelope = layer["cross_variant"]["envelope"]
        if not envelope["all_finite"]:
            failures.append(
                {"layer_index": layer["layer_index"], "reason": "nonfinite"}
            )
        if not envelope["all_within_tolerance"]:
            failures.append(
                {"layer_index": layer["layer_index"], "reason": "outside_atol_rtol"}
            )
        if envelope["min_cosine"] < MODEL_QUALIFICATION_THRESHOLDS["cosine_min"]:
            failures.append(
                {"layer_index": layer["layer_index"], "reason": "cosine_below_minimum"}
            )
        if envelope["max_mae"] > MODEL_QUALIFICATION_THRESHOLDS["mae_max"]:
            failures.append(
                {"layer_index": layer["layer_index"], "reason": "mae_above_maximum"}
            )
        for variant in ("reference", "candidate"):
            if not layer["repeatability"][variant]["envelope"]["all_exact"]:
                failures.append(
                    {
                        "layer_index": layer["layer_index"],
                        "variant": variant,
                        "reason": "repeatability_not_bitwise",
                    }
                )
    if reference_hits != [0] * 5 or candidate_hits != [5] * 5:
        failures.append({"reason": "direct_hit_count_mismatch"})
    if any(len(set(pointers)) != 1 for pointers in candidate_pointers.values()):
        failures.append({"reason": "caller_output_not_reused"})

    report = {
        "comparison_mode": "correctness",
        "run_order": args.run_order,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "component_name": component_name,
        "capture_manifest_path": str(capture_path),
        "capture_manifest_sha256": manifest["manifest_sha256"],
        "capture_coordinates": coordinates,
        "candidate_route": config,
        "production_routed_candidate": actual_timestep <= 521,
        "port_provenance": port_provenance,
        "qkv_contract": {
            "layout": "NHD",
            "shape": [1, 4800, 40, 128],
            "dtype": "torch.bfloat16",
            "post_rope": True,
        },
        "reference_per_run_wan_hybrid_hit_count": reference_hits,
        "candidate_per_run_wan_hybrid_hit_count": candidate_hits,
        "candidate_per_run_wan_hybrid_expected_hit_count": [5] * 5,
        "candidate_caller_output_data_ptr": candidate_pointers,
        "layers": layer_reports,
        "qualification": {
            "passed": not failures,
            "thresholds": MODEL_QUALIFICATION_THRESHOLDS,
            "failures": failures,
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Wan direct attention qualification failed: {failures}")


if __name__ == "__main__":
    main()
