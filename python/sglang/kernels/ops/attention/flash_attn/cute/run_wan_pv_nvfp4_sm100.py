"""Runtime smoke for the exact Wan BF16-QK + P1/V2 SM100 path."""

import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100")

import torch

from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _WAN_PV_NVFP4_PACKED_SHAPE,
    _WAN_PV_NVFP4_SF_NUMEL,
    _WAN_PV_NVFP4_SHAPE,
    _flash_attn_fwd,
)


def _launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    v_base: torch.Tensor,
    v_residual: torch.Tensor,
    sfv_base: torch.Tensor,
    sfv_residual: torch.Tensor,
) -> None:
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
    if returned_out.data_ptr() != out.data_ptr():
        raise RuntimeError("pv_nvfp4 did not return the caller-owned output")
    if returned_lse is not None:
        raise RuntimeError("pv_nvfp4 unexpectedly materialized LSE")


def _diagnose_case(
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    v_base: torch.Tensor,
    v_residual: torch.Tensor,
    sfv_base: torch.Tensor,
    sfv_residual: torch.Tensor,
) -> None:
    first_sentinel = 1232.0
    second_sentinel = -976.0

    out.fill_(first_sentinel)
    _launch(q, k, v, out, v_base, v_residual, sfv_base, sfv_residual)
    torch.cuda.synchronize()
    first = out.clone()

    out.fill_(second_sentinel)
    _launch(q, k, v, out, v_base, v_residual, sfv_base, sfv_residual)
    torch.cuda.synchronize()

    first_changed = first != first_sentinel
    second_changed = out != second_sentinel
    both_changed = first_changed & second_changed
    same_bits = first.view(torch.int16) == out.view(torch.int16)
    repeatable_changed = both_changed & same_bits
    first_finite = torch.isfinite(first)
    second_finite = torch.isfinite(out)
    first_changed_count = int(first_changed.sum().item())
    second_changed_count = int(second_changed.sum().item())
    both_changed_count = int(both_changed.sum().item())
    repeatable_changed_count = int(repeatable_changed.sum().item())
    first_zero_count = int(((first == 0) & first_changed).sum().item())
    second_zero_count = int(((out == 0) & second_changed).sum().item())

    # Shape is BSHD. Summarize which query tokens, heads, and output dimensions
    # were changed without dumping a 24.6-million-element mask.
    changed_per_token = second_changed.sum(dim=(2, 3))
    changed_per_head = second_changed.sum(dim=(0, 1, 3))
    changed_per_dim = second_changed.sum(dim=(0, 1, 2))
    token_any_count = int((changed_per_token > 0).sum().item())
    token_full_count = int(
        (changed_per_token == q.shape[2] * q.shape[3]).sum().item()
    )

    print(
        f"runtime exact Wan pv_nvfp4 SM100 diagnostics ({name}): "
        f"numel={out.numel()} first_changed={first_changed_count} "
        f"second_changed={second_changed_count} both_changed={both_changed_count} "
        f"repeatable_changed={repeatable_changed_count} "
        f"first_nan={int(torch.isnan(first).sum().item())} "
        f"second_nan={int(torch.isnan(out).sum().item())} "
        f"first_nonfinite={int((~first_finite).sum().item())} "
        f"second_nonfinite={int((~second_finite).sum().item())} "
        f"first_changed_zero={first_zero_count} "
        f"second_changed_zero={second_zero_count} "
        f"token_any={token_any_count}/{q.shape[0] * q.shape[1]} "
        f"token_full={token_full_count}/{q.shape[0] * q.shape[1]} "
        f"changed_per_token_minmax="
        f"{int(changed_per_token.min().item())}/{int(changed_per_token.max().item())} "
        f"changed_per_head_minmax="
        f"{int(changed_per_head.min().item())}/{int(changed_per_head.max().item())} "
        f"changed_per_dim_minmax="
        f"{int(changed_per_dim.min().item())}/{int(changed_per_dim.max().item())}",
        flush=True,
    )
    if second_changed_count != out.numel():
        raise RuntimeError(f"{name}: pv_nvfp4 did not materialize the full output")
    if int((~second_finite).sum().item()) != 0:
        raise RuntimeError(f"{name}: pv_nvfp4 output contains non-finite values")
    if repeatable_changed_count != out.numel():
        raise RuntimeError(f"{name}: pv_nvfp4 repeated launch was not bitwise repeatable")
    if second_zero_count != out.numel():
        raise RuntimeError(f"{name}: zero V workspaces did not produce all-zero output")


def main() -> None:
    torch.manual_seed(20260819)
    q = torch.randn(_WAN_PV_NVFP4_SHAPE, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.zeros_like(q)
    out = torch.empty_like(q)
    v_base = torch.zeros(
        _WAN_PV_NVFP4_PACKED_SHAPE, dtype=torch.uint8, device="cuda"
    )
    v_residual = torch.zeros_like(v_base)
    sfv_base = torch.ones(
        (_WAN_PV_NVFP4_SF_NUMEL,),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    sfv_residual = torch.ones_like(sfv_base)

    _diagnose_case(
        "zero_qk",
        torch.zeros_like(q),
        torch.zeros_like(k),
        v,
        out,
        v_base,
        v_residual,
        sfv_base,
        sfv_residual,
    )
    _diagnose_case(
        "random_qk",
        q,
        k,
        v,
        out,
        v_base,
        v_residual,
        sfv_base,
        sfv_residual,
    )
    print("runtime exact Wan pv_nvfp4 SM100 zero-V forward passed")


if __name__ == "__main__":
    main()
