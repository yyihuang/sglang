"""Paired cold-L2 CUPTI benchmark for exact Wan BF16-QK + P1/V2 SM100.

``C`` denotes the candidate and ``F`` denotes the production FA4 default path.
Each candidate invocation starts from raw BF16 V and includes padded-row packing,
base/residual FP4 quantization, copies into reusable packed/scales workspaces, the
``pv_nvfp4`` attention launch, and direct materialization into a caller-owned BF16
output.  FlashInfer's current ``fp4_quantize`` API has no ``out=`` parameter, so
its returned packed and scale tensors remain temporary allocations inside the
timed candidate invocation; the emitted report records that limitation.
"""

import inspect
import json
import os
import statistics
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import Callable

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch
from flashinfer import fp4_quantize
from flashinfer.testing import bench_gpu_time

from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _WAN_PV_NVFP4_PACKED_SHAPE,
    _WAN_PV_NVFP4_SF_NUMEL,
    _WAN_PV_NVFP4_SHAPE,
    _flash_attn_fwd,
)
from sglang.test.quant_ref_utils import dequantize_nvfp4_to_dtype


_PADDED_SEQUENCE = _WAN_PV_NVFP4_PACKED_SHAPE[-1] * 2
_V_ROWS = (
    _WAN_PV_NVFP4_SHAPE[0]
    * _WAN_PV_NVFP4_SHAPE[2]
    * _WAN_PV_NVFP4_SHAPE[3]
)
_WARMUP_RUNS = 2
_MEASURE_RUNS = 5
_PAIRED_ORDERS = (("C", "F", "F", "C"), ("F", "C", "C", "F"))


@dataclass
class CandidateWorkspace:
    """Caller-owned storage reused by every candidate invocation."""

    rows: torch.Tensor
    residual_f32: torch.Tensor
    residual_bf16: torch.Tensor
    v_base: torch.Tensor
    v_residual: torch.Tensor
    sfv_base: torch.Tensor
    sfv_residual: torch.Tensor
    global_scale: torch.Tensor


def _require_cupti() -> str:
    """Fail closed instead of letting ``bench_gpu_time`` use event fallback."""

    try:
        from cupti import cupti as _cupti  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("cupti-python is required for this benchmark") from error
    cupti_version = distribution_version("cupti-python")
    if int(cupti_version.split(".", maxsplit=1)[0]) < 13:
        raise RuntimeError(
            f"cupti-python>=13 is required, found {cupti_version}"
        )
    return cupti_version


def _allocate_candidate_workspace(v: torch.Tensor) -> CandidateWorkspace:
    device = v.device
    return CandidateWorkspace(
        rows=torch.empty((_V_ROWS, _PADDED_SEQUENCE), dtype=v.dtype, device=device),
        residual_f32=torch.empty(
            (_V_ROWS, _PADDED_SEQUENCE), dtype=torch.float32, device=device
        ),
        residual_bf16=torch.empty(
            (_V_ROWS, _PADDED_SEQUENCE), dtype=v.dtype, device=device
        ),
        v_base=torch.empty(
            _WAN_PV_NVFP4_PACKED_SHAPE, dtype=torch.uint8, device=device
        ),
        v_residual=torch.empty(
            _WAN_PV_NVFP4_PACKED_SHAPE, dtype=torch.uint8, device=device
        ),
        sfv_base=torch.empty(
            (_WAN_PV_NVFP4_SF_NUMEL,),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        sfv_residual=torch.empty(
            (_WAN_PV_NVFP4_SF_NUMEL,),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        global_scale=torch.ones((1,), dtype=torch.float32, device=device),
    )


def _quantize_v2_into(
    v_rows_source: torch.Tensor, workspace: CandidateWorkspace
) -> None:
    """Quantize raw BF16 V into the exact P1/V2 caller-owned workspace."""

    workspace.rows.zero_()
    workspace.rows[:, : _WAN_PV_NVFP4_SHAPE[1]].copy_(v_rows_source)

    base_tmp, sf_base_tmp = fp4_quantize(
        workspace.rows,
        workspace.global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    base_tmp_u8 = base_tmp.view(torch.uint8).reshape(
        _V_ROWS, _PADDED_SEQUENCE // 2
    )
    workspace.v_base.copy_(
        base_tmp_u8.reshape(_WAN_PV_NVFP4_PACKED_SHAPE)
    )
    workspace.sfv_base.copy_(
        sf_base_tmp.view(torch.float8_e4m3fn).reshape(-1)
    )

    # Match the numerical qualifier's FP32 base reconstruction and subtraction,
    # while keeping both subtraction destinations reusable.  The independent
    # reference dequantizer itself has no out= API and therefore still creates
    # temporary tensors inside this timed candidate invocation.
    base_dequant_tmp = dequantize_nvfp4_to_dtype(
        base_tmp_u8, sf_base_tmp, 1.0, torch.float32
    )
    workspace.residual_f32.copy_(workspace.rows)
    torch.sub(workspace.residual_f32, base_dequant_tmp, out=workspace.residual_f32)
    workspace.residual_bf16.copy_(workspace.residual_f32)

    residual_tmp, sf_residual_tmp = fp4_quantize(
        workspace.residual_bf16,
        workspace.global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    workspace.v_residual.copy_(
        residual_tmp.view(torch.uint8).reshape(_WAN_PV_NVFP4_PACKED_SHAPE)
    )
    workspace.sfv_residual.copy_(
        sf_residual_tmp.view(torch.float8_e4m3fn).reshape(-1)
    )


def _production_fa4(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
) -> None:
    returned_out, returned_lse = _flash_attn_fwd(
        q,
        k,
        v,
        out=out,
        _arch=100,
        pack_gqa=False,
    )
    if returned_out.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise RuntimeError("production FA4 did not honor the caller-owned output ABI")


def _candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_rows_source: torch.Tensor,
    out: torch.Tensor,
    workspace: CandidateWorkspace,
) -> None:
    _quantize_v2_into(v_rows_source, workspace)
    returned_out, returned_lse = _flash_attn_fwd(
        q,
        k,
        v,
        out=out,
        _arch=100,
        pack_gqa=False,
        pv_nvfp4=True,
        v_base=workspace.v_base,
        v_residual=workspace.v_residual,
        sfv_base=workspace.sfv_base,
        sfv_residual=workspace.sfv_residual,
    )
    if returned_out.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise RuntimeError("candidate did not honor the caller-owned output ABI")


def _candidate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace: CandidateWorkspace,
    *,
    pv_nvfp4_residual: bool = True,
) -> None:
    """Run only attention against an already prepared V2 workspace."""

    returned_out, returned_lse = _flash_attn_fwd(
        q,
        k,
        v,
        out=out,
        _arch=100,
        pack_gqa=False,
        pv_nvfp4=True,
        pv_nvfp4_residual=pv_nvfp4_residual,
        v_base=workspace.v_base,
        v_residual=workspace.v_residual,
        sfv_base=workspace.sfv_base,
        sfv_residual=workspace.sfv_residual,
    )
    if returned_out.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise RuntimeError("candidate did not honor the caller-owned output ABI")


def _measure_leg(fn: Callable[[], None]) -> list[float]:
    return [
        float(sample)
        for sample in bench_gpu_time(
            fn=fn,
            dry_run_iters=_WARMUP_RUNS,
            repeat_iters=_MEASURE_RUNS,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
    ]


def _measure_order(
    order: tuple[str, str, str, str],
    candidate_fn: Callable[[], None],
    fa4_fn: Callable[[], None],
) -> dict:
    functions = {"C": candidate_fn, "F": fa4_fn}
    legs = []
    pooled = {"C": [], "F": []}
    for leg_index, label in enumerate(order):
        samples = _measure_leg(functions[label])
        pooled[label].extend(samples)
        legs.append(
            {
                "leg": leg_index,
                "provider": "candidate" if label == "C" else "production_fa4",
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )

    candidate_ms = statistics.median(pooled["C"])
    production_fa4_ms = statistics.median(pooled["F"])
    speedup = production_fa4_ms / candidate_ms
    return {
        "order": "/".join(order),
        "legs": legs,
        "candidate_median_ms": candidate_ms,
        "production_fa4_median_ms": production_fa4_ms,
        "speedup": speedup,
        "passed_speedup_ge_1": speedup >= 1.0,
    }


def _measure_components(
    quantize_fn: Callable[[], None],
    attention_fn: Callable[[], None],
    fa4_fn: Callable[[], None],
) -> dict:
    """Attribute the complete boundary without treating parts as a speedup."""

    functions = {
        "quantizer": quantize_fn,
        "attention": attention_fn,
        "production_fa4": fa4_fn,
    }
    order = (
        "quantizer",
        "attention",
        "production_fa4",
        "production_fa4",
        "attention",
        "quantizer",
    )
    pooled = {name: [] for name in functions}
    legs = []
    for leg_index, name in enumerate(order):
        samples = _measure_leg(functions[name])
        pooled[name].extend(samples)
        legs.append(
            {
                "leg": leg_index,
                "component": name,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
    medians = {name: statistics.median(samples) for name, samples in pooled.items()}
    return {
        "diagnostic_only": True,
        "order": "/".join(order),
        "legs": legs,
        "pooled_median_ms": medians,
        "quantizer_plus_attention_ms": medians["quantizer"] + medians["attention"],
    }


def main() -> None:
    cupti_version = _require_cupti()
    if "out" in inspect.signature(fp4_quantize).parameters:
        raise RuntimeError(
            "fp4_quantize now exposes out=; update this benchmark to use it before "
            "claiming reusable quantizer outputs"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(4254)
    q = torch.randn(
        _WAN_PV_NVFP4_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        _WAN_PV_NVFP4_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        _WAN_PV_NVFP4_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    out_candidate = torch.empty_like(q)
    out_fa4 = torch.empty_like(q)
    workspace = _allocate_candidate_workspace(v)

    # This is a strided view over raw BF16 NHD V.  Every candidate invocation
    # performs the GPU copy into its padded reusable row-major staging buffer.
    v_rows_source = v.permute(0, 2, 3, 1).reshape(
        _V_ROWS, _WAN_PV_NVFP4_SHAPE[1]
    )
    candidate_fn = lambda: _candidate(
        q, k, v, v_rows_source, out_candidate, workspace
    )
    quantize_fn = lambda: _quantize_v2_into(v_rows_source, workspace)
    attention_fn = lambda: _candidate_attention(
        q, k, v, out_candidate, workspace
    )
    fa4_fn = lambda: _production_fa4(q, k, v, out_fa4)

    # Materialize JIT kernels and allocator pools before either paired order.
    # Per-leg warmup and the measured candidate body still include all V2 work.
    fa4_fn()
    candidate_fn()
    torch.cuda.synchronize()

    components = _measure_components(quantize_fn, attention_fn, fa4_fn)
    orders = [
        _measure_order(order, candidate_fn, fa4_fn) for order in _PAIRED_ORDERS
    ]
    passed = all(result["passed_speedup_ge_1"] for result in orders)
    report = {
        "passed": passed,
        "legend": {"C": "candidate", "F": "production_fa4"},
        "shape_bshd": list(_WAN_PV_NVFP4_SHAPE),
        "logical_layout": "NHD",
        "causal": False,
        "dtype": "bfloat16",
        "seed": 4254,
        "device": torch.cuda.get_device_name(q.device),
        "stream": int(torch.cuda.current_stream(q.device).cuda_stream),
        "timing": {
            "backend": "CUPTI activity span",
            "cold_l2": True,
            "cuda_graph": False,
            "warmup_runs_per_leg": _WARMUP_RUNS,
            "measure_runs_per_leg": _MEASURE_RUNS,
        },
        "candidate_scope": [
            "raw BF16 V to padded row-major staging",
            "base FP4 quantize and pack",
            "FP32 base dequantize and residual construction",
            "residual FP4 quantize and pack",
            "pv_nvfp4 attention",
            "direct caller-owned BF16 output materialization",
        ],
        "workspace": {
            "reused": [
                "padded_v_rows",
                "residual_f32",
                "residual_bf16",
                "v_base",
                "v_residual",
                "sfv_base",
                "sfv_residual",
                "candidate_output",
                "production_fa4_output",
            ],
            "fp4_quantize_supports_out": False,
            "temporary_allocation_gap": (
                "flashinfer.fp4_quantize has no out= parameter, so both timed "
                "quantize calls return temporary packed/scales tensors before "
                "copying them into reusable caller-owned workspaces; the "
                "reference dequantizer also returns temporary tensors"
            ),
        },
        "cupti_python_version": cupti_version,
        "components": components,
        "orders": orders,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        failed_orders = [
            result["order"]
            for result in orders
            if not result["passed_speedup_ge_1"]
        ]
        raise RuntimeError(
            "exact Wan paired performance gate failed for "
            + ", ".join(failed_orders)
        )


if __name__ == "__main__":
    main()
