"""Numerical oracle for the exact Wan BF16-QK + P1/V2 NVFP4 contract.

This helper models the intended SM100 arithmetic without importing or launching
the experimental attention kernel.  Q and K remain BF16, P is quantized in
per-16 E2M1 blocks with E4M3 scales, and V uses the two-level base/residual
NVFP4 representation consumed by the kernel.
"""

import json
import math
import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize

from sglang.test.quant_ref_utils import dequantize_nvfp4_to_dtype


_WAN_SHAPE = (1, 4800, 40, 128)
_PADDED_SEQUENCE = 4864
_SF_VEC_SIZE = 16
_P_QUANT_MULTIPLIER = 512.0
_E4M3_SMALLEST_SUBNORMAL = 2.0**-9
_Q_BLOCK_SIZE = 128
_SEED = 4254


def _quantize_dequantize_v2(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dequantized V base and residual in contiguous HSD layout."""
    batch, sequence, heads, head_dim = v.shape
    rows_count = batch * heads * head_dim
    rows = torch.zeros(
        (rows_count, _PADDED_SEQUENCE), dtype=v.dtype, device=v.device
    )
    rows[:, :sequence].copy_(
        v.permute(0, 2, 3, 1).reshape(rows_count, sequence)
    )
    global_scale = torch.ones((1,), dtype=torch.float32, device=v.device)

    base, sf_base = fp4_quantize(
        rows,
        global_scale,
        sf_vec_size=_SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    base_u8 = base.view(torch.uint8).reshape(
        rows_count, _PADDED_SEQUENCE // 2
    )
    base_dequant = dequantize_nvfp4_to_dtype(
        base_u8, sf_base, 1.0, torch.float32
    )

    # Match the V2 producer: form the residual from the FP32 base
    # reconstruction, then round that residual source back to BF16.
    residual_source = (rows.float() - base_dequant).to(v.dtype)
    residual, sf_residual = fp4_quantize(
        residual_source,
        global_scale,
        sf_vec_size=_SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    residual_u8 = residual.view(torch.uint8).reshape(
        rows_count, _PADDED_SEQUENCE // 2
    )
    residual_dequant = dequantize_nvfp4_to_dtype(
        residual_u8, sf_residual, 1.0, torch.float32
    )

    def rows_to_hsd(rows_f32: torch.Tensor) -> torch.Tensor:
        return (
            rows_f32[:, :sequence]
            .reshape(batch, heads, head_dim, sequence)
            .permute(0, 1, 3, 2)
            .contiguous()[0]
        )

    return rows_to_hsd(base_dequant), rows_to_hsd(residual_dequant)


def _round_nonnegative_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round nonnegative values to {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    rounded = torch.zeros_like(values)
    # Halfway cases follow round-to-nearest-even in the E2M1 encoding.
    rounded.masked_fill_(values > 0.25, 0.5)
    rounded.masked_fill_(values >= 0.75, 1.0)
    rounded.masked_fill_(values > 1.25, 1.5)
    rounded.masked_fill_(values >= 1.75, 2.0)
    rounded.masked_fill_(values > 2.5, 3.0)
    rounded.masked_fill_(values >= 3.5, 4.0)
    rounded.masked_fill_(values > 5.0, 6.0)
    return rounded


def _quantize_dequantize_p(p: torch.Tensor) -> torch.Tensor:
    """Model the core's fixed-512, per-16 E4M3/E2M1 P round trip."""
    if p.dtype != torch.float32 or not p.is_contiguous():
        raise ValueError("P must be contiguous FP32")
    if p.shape[-1] % _SF_VEC_SIZE:
        raise ValueError("P's sequence dimension must be divisible by 16")

    blocks = (p * _P_QUANT_MULTIPLIER).reshape(
        *p.shape[:-1], p.shape[-1] // _SF_VEC_SIZE, _SF_VEC_SIZE
    )
    amax = blocks.amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / 6.0)).to(torch.float8_e4m3fn).float()

    # This mirrors the kernel's finite reciprocal for underflowing E4M3
    # scales.  A zero scale forces the corresponding FP4 payload to zero.
    denominator = scale.clamp_min(_E4M3_SMALLEST_SUBNORMAL)
    normalized = blocks / denominator
    normalized.masked_fill_(scale == 0.0, 0.0)
    fp4 = _round_nonnegative_e2m1(normalized)
    return (fp4 * scale / _P_QUANT_MULTIPLIER).reshape_as(p)


def _oracle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v_base: torch.Tensor,
    v_residual: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact shape in bounded Q blocks without a candidate call."""
    q_hsd = q[0].permute(1, 0, 2).contiguous()
    k_hds = k[0].permute(1, 2, 0).contiguous().float()
    output = torch.empty_like(q_hsd)
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    for q_start in range(0, q.shape[1], _Q_BLOCK_SIZE):
        q_stop = min(q_start + _Q_BLOCK_SIZE, q.shape[1])
        scores = torch.bmm(q_hsd[:, q_start:q_stop].float(), k_hds)
        p = torch.softmax(scores * softmax_scale, dim=-1).contiguous()
        p_nvfp4 = _quantize_dequantize_p(p)
        out_f32 = torch.bmm(p_nvfp4, v_base)
        out_f32.add_(torch.bmm(p_nvfp4, v_residual))
        output[:, q_start:q_stop].copy_(out_f32.to(torch.bfloat16))

    return output.permute(1, 0, 2).unsqueeze(0).contiguous()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    return {
        "finite": bool(torch.isfinite(actual_f32).all().item()),
        "reference_finite": bool(torch.isfinite(expected_f32).all().item()),
        "atol_1_rtol_0_1": bool(
            torch.allclose(actual_f32, expected_f32, atol=1.0, rtol=0.1)
        ),
        "cosine": float(
            F.cosine_similarity(
                actual_f32.flatten(), expected_f32.flatten(), dim=0
            ).item()
        ),
        "mae": float(difference.abs().mean().item()),
        "max_abs_error": float(difference.abs().max().item()),
    }


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the exact Wan NVFP4 oracle requires a CUDA GPU")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(_SEED)
    q = torch.randn(
        _WAN_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        _WAN_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        _WAN_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    v_base, v_residual = _quantize_dequantize_v2(v)
    first = _oracle_attention(q, k, v_base, v_residual)
    second = _oracle_attention(q, k, v_base, v_residual)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(_WAN_SHAPE[-1]),
    ).transpose(1, 2).contiguous()
    torch.cuda.synchronize()

    metrics = _metrics(second, reference)
    repeatability = {
        "bitwise": bool(torch.equal(first, second)),
        "max_abs_error": float(
            (first.float() - second.float()).abs().max().item()
        ),
    }
    passed = bool(
        metrics["finite"]
        and metrics["reference_finite"]
        and metrics["atol_1_rtol_0_1"]
        and metrics["cosine"] >= 0.995
        and metrics["mae"] <= 0.025
        and repeatability["bitwise"]
    )
    result = {
        "contract": {
            "shape": list(_WAN_SHAPE),
            "layout": "NHD",
            "causal": False,
            "qk_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "p_format": "per-16 E2M1 with E4M3 scale",
            "p_quant_multiplier": _P_QUANT_MULTIPLIER,
            "v_format": "V2 base+residual per-16 E2M1 with E4M3 scale",
        },
        "seed": _SEED,
        "oracle_vs_torch_bf16_attention": metrics,
        "repeatability": repeatability,
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("exact Wan P+V numerical oracle failed")


if __name__ == "__main__":
    main()
