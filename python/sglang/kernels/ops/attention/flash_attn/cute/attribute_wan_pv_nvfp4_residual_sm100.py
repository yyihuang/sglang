"""B200 attribution for the second NVFP4 V level in exact Wan attention."""

import json
import math
import os
import statistics

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.flash_attn.cute.benchmark_wan_pv_nvfp4_paired_sm100 import (
    _V_ROWS,
    _WAN_PV_NVFP4_SHAPE,
    _allocate_candidate_workspace,
    _candidate_attention,
    _measure_leg,
    _production_fa4,
    _quantize_v2_into,
)
from sglang.kernels.ops.attention.flash_attn.cute.qualify_wan_pv_nvfp4_sm100 import (
    _metrics,
)


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
    workspace = _allocate_candidate_workspace(v)
    v_rows_source = v.permute(0, 2, 3, 1).reshape(
        _V_ROWS, _WAN_PV_NVFP4_SHAPE[1]
    )
    _quantize_v2_into(v_rows_source, workspace)

    out_base = torch.empty_like(q)
    out_full = torch.empty_like(q)
    out_fa4 = torch.empty_like(q)
    base_fn = lambda: _candidate_attention(
        q,
        k,
        v,
        out_base,
        workspace,
        pv_nvfp4_residual=False,
    )
    full_fn = lambda: _candidate_attention(q, k, v, out_full, workspace)
    fa4_fn = lambda: _production_fa4(q, k, v, out_fa4)

    # Materialize all three specializations before attribution.
    fa4_fn()
    full_fn()
    base_fn()
    torch.cuda.synchronize()

    base_first = out_base.clone()
    base_fn()
    torch.cuda.synchronize()
    base_repeatable = bool(torch.equal(base_first, out_base))

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

    order = (
        "base_only",
        "full_v2",
        "production_fa4",
        "production_fa4",
        "full_v2",
        "base_only",
    )
    functions = {
        "base_only": base_fn,
        "full_v2": full_fn,
        "production_fa4": fa4_fn,
    }
    pooled = {name: [] for name in functions}
    legs = []
    for leg_index, name in enumerate(order):
        samples = _measure_leg(functions[name])
        pooled[name].extend(samples)
        legs.append(
            {
                "leg": leg_index,
                "provider": name,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
    medians = {name: statistics.median(samples) for name, samples in pooled.items()}

    report = {
        "diagnostic_only": True,
        "shape_bshd": list(q.shape),
        "layout": "NHD",
        "seed": 4254,
        "device": torch.cuda.get_device_name(q.device),
        "timing": {
            "backend": "CUPTI activity span",
            "cold_l2": True,
            "warmup_runs_per_leg": 2,
            "measure_runs_per_leg": 5,
            "order": list(order),
            "legs": legs,
            "pooled_median_ms": medians,
            "residual_cost_ms": medians["full_v2"] - medians["base_only"],
        },
        "correctness": {
            "base_only_vs_torch": _metrics(out_base, out_torch),
            "full_v2_vs_torch": _metrics(out_full, out_torch),
            "production_fa4_vs_torch": _metrics(out_fa4, out_torch),
            "base_only_repeatable_bitwise": base_repeatable,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
