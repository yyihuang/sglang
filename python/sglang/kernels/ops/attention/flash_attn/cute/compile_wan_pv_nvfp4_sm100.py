"""Compile-only smoke for the exact Wan BF16-QK + P1/V2 SM100 path."""

import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _WAN_PV_NVFP4_PACKED_SHAPE,
    _WAN_PV_NVFP4_SF_NUMEL,
    _WAN_PV_NVFP4_SHAPE,
    _flash_attn_fwd,
)


def main() -> None:
    with FakeTensorMode():
        q = torch.empty(_WAN_PV_NVFP4_SHAPE, dtype=torch.bfloat16, device="cuda")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        out = torch.empty_like(q)
        v_base = torch.empty(
            _WAN_PV_NVFP4_PACKED_SHAPE, dtype=torch.uint8, device="cuda"
        )
        v_residual = torch.empty_like(v_base)
        sfv_base = torch.empty(
            (_WAN_PV_NVFP4_SF_NUMEL,),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        sfv_residual = torch.empty_like(sfv_base)
        _flash_attn_fwd(
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
    print("compiled exact Wan pv_nvfp4 SM100 forward")


if __name__ == "__main__":
    main()
