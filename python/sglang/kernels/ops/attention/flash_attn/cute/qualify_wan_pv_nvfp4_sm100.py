"""Exact-shape numerical qualification for Wan BF16-QK + P1/V2 SM100."""

import json
import math
import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize

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


def _quantize_v2(v: torch.Tensor):
    rows = torch.zeros(
        (_V_ROWS, _PADDED_SEQUENCE), dtype=v.dtype, device=v.device
    )
    rows[:, : v.shape[1]].copy_(v.permute(0, 2, 3, 1).reshape(_V_ROWS, -1))
    global_scale = torch.ones((1,), dtype=torch.float32, device=v.device)

    base, sf_base = fp4_quantize(
        rows,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    base_u8 = base.view(torch.uint8).reshape(_V_ROWS, _PADDED_SEQUENCE // 2)
    base_dequant = dequantize_nvfp4_to_dtype(
        base_u8, sf_base, 1.0, torch.float32
    )
    residual_source = (rows.float() - base_dequant).to(v.dtype)
    residual, sf_residual = fp4_quantize(
        residual_source,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )

    v_base = base_u8.reshape(_WAN_PV_NVFP4_PACKED_SHAPE).contiguous()
    v_residual = (
        residual.view(torch.uint8)
        .reshape(_WAN_PV_NVFP4_PACKED_SHAPE)
        .contiguous()
    )
    sfv_base = sf_base.view(torch.float8_e4m3fn).reshape(-1).contiguous()
    sfv_residual = (
        sf_residual.view(torch.float8_e4m3fn).reshape(-1).contiguous()
    )
    if sfv_base.numel() != _WAN_PV_NVFP4_SF_NUMEL:
        raise RuntimeError(
            f"base scale numel={sfv_base.numel()}, "
            f"expected {_WAN_PV_NVFP4_SF_NUMEL}"
        )
    if sfv_residual.numel() != _WAN_PV_NVFP4_SF_NUMEL:
        raise RuntimeError(
            f"residual scale numel={sfv_residual.numel()}, "
            f"expected {_WAN_PV_NVFP4_SF_NUMEL}"
        )
    return v_base, v_residual, sfv_base, sfv_residual


def _production_fa4(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
) -> None:
    returned_out, returned_lse = _flash_attn_fwd(
        q, k, v, out=out, _arch=100, pack_gqa=False
    )
    if returned_out.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise RuntimeError("production FA4 did not honor the dense caller-owned ABI")


def _candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspaces,
) -> None:
    v_base, v_residual, sfv_base, sfv_residual = workspaces
    returned_out, returned_lse = _flash_attn_fwd(
        q,
        k,
        v,
        out=out,
        _arch=100,
        pack_gqa=False,
        pv_nvfp4=True,
        v_base=v_base,
        v_residual=v_residual,
        sfv_base=sfv_base,
        sfv_residual=sfv_residual,
    )
    if returned_out.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise RuntimeError("candidate did not honor the dense caller-owned ABI")


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = actual_f32 - expected_f32
    cosine = float(
        F.cosine_similarity(actual_f32.flatten(), expected_f32.flatten(), dim=0)
        .item()
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "atol_1_rtol_0_1": bool(
            torch.allclose(actual_f32, expected_f32, atol=1.0, rtol=0.1)
        ),
        "cosine": cosine,
        "mae": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
    }


def main() -> None:
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
    out_fa4 = torch.empty_like(q)
    out_candidate = torch.empty_like(q)

    workspaces = _quantize_v2(v)
    _production_fa4(q, k, v, out_fa4)
    _candidate(q, k, v, out_candidate, workspaces)
    torch.cuda.synchronize()
    first_candidate = out_candidate.clone()
    _candidate(q, k, v, out_candidate, workspaces)
    torch.cuda.synchronize()

    q_hnd = q.transpose(1, 2)
    k_hnd = k.transpose(1, 2)
    v_hnd = v.transpose(1, 2)
    out_torch = F.scaled_dot_product_attention(
        q_hnd,
        k_hnd,
        v_hnd,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(q.shape[-1]),
    ).transpose(1, 2).contiguous()
    torch.cuda.synchronize()

    result = {
        "shape": list(q.shape),
        "seed": 4254,
        "candidate_vs_torch": _metrics(out_candidate, out_torch),
        "candidate_vs_production_fa4": _metrics(out_candidate, out_fa4),
        "production_fa4_vs_torch": _metrics(out_fa4, out_torch),
        "repeatable_bitwise": bool(torch.equal(first_candidate, out_candidate)),
    }
    candidate_metrics = result["candidate_vs_torch"]
    passed = (
        candidate_metrics["finite"]
        and candidate_metrics["atol_1_rtol_0_1"]
        and candidate_metrics["cosine"] >= 0.995
        and candidate_metrics["mae"] <= 0.025
        and result["repeatable_bitwise"]
    )
    result["passed"] = passed
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("exact Wan pv_nvfp4 numerical qualification failed")


if __name__ == "__main__":
    main()
