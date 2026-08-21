"""Single-launch Nsight Compute driver for the exact Wan hybrid attention core.

This diagnostic prepares the two-level FP4 V workspace before profiling, then
launches only the already-packed hybrid attention path.  Quantization is kept
out of the profiled launch so the report attributes the structural attention
gap rather than the known end-to-end workspace/API overhead.
"""

import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch

from sglang.kernels.ops.attention.flash_attn.cute.benchmark_wan_pv_nvfp4_paired_sm100 import (
    _V_ROWS,
    _WAN_PV_NVFP4_SHAPE,
    _allocate_candidate_workspace,
    _candidate_attention,
    _production_fa4,
    _quantize_v2_into,
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
    out = torch.empty_like(q)
    mode = os.environ.get("WAN_NVFP4_PROFILE_MODE", "candidate")
    if mode in ("candidate", "candidate_base_only"):
        workspace = _allocate_candidate_workspace(v)
        v_rows_source = v.permute(0, 2, 3, 1).reshape(
            _V_ROWS, _WAN_PV_NVFP4_SHAPE[1]
        )
        _quantize_v2_into(v_rows_source, workspace)
        pv_nvfp4_residual = mode == "candidate"
        launch = lambda: _candidate_attention(
            q,
            k,
            v,
            out,
            workspace,
            pv_nvfp4_residual=pv_nvfp4_residual,
        )
    elif mode == "production_fa4":
        launch = lambda: _production_fa4(q, k, v, out)
    else:
        raise ValueError(f"unsupported WAN_NVFP4_PROFILE_MODE={mode!r}")

    # First call materializes the JIT specialization.  Nsight Compute should
    # select the second matching FlashAttentionForwardSm100 launch.
    launch()
    torch.cuda.synchronize()
    print(f"WAN_NVFP4_PROFILE_TARGET_BEGIN mode={mode}", flush=True)
    launch()
    torch.cuda.synchronize()
    print(f"WAN_NVFP4_PROFILE_TARGET_END mode={mode}", flush=True)


if __name__ == "__main__":
    main()
