"""CAKE KDA prefill through the exported prepared BF16 API (FP32 state pool).

The prepared export (``flashinfer.prepare_bf16_kda_prefill``) is selected by the
Cake kernel whenever the recurrent state pool is FP32. These tests compare it
against SGLang's Triton KDA prefill on the same FP32 pool.
"""

import os
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="4-gpu-b200",
    disabled="prepared BF16 KDA export is not in the pinned public FlashInfer build",
)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
):
    pytest.skip("CAKE KDA export requires SM100/SM103", allow_module_level=True)

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (  # noqa: E402
    CakeKDAKernel,
    _get_flashinfer_prepared_bf16_prefill,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)

if not _get_flashinfer_prepared_bf16_prefill()[0]:
    pytest.skip(
        "installed FlashInfer lacks prepare_bf16_kda_prefill", allow_module_level=True
    )

K = V = 128
LOWER_BOUND = -5.0


def _make_inputs(seq_lens, num_heads, *, state_dtype=torch.float32):
    num_sequences = len(seq_lens)
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    pool_size = num_sequences + 5
    cache_indices = torch.arange(
        pool_size - 1,
        pool_size - num_sequences - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    make = lambda: torch.randn(  # noqa: E731
        1, total_tokens, num_heads, K, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    return dict(
        q=make(),
        k=make(),
        v=make(),
        g=make(),
        # Fused-projection slice: raw BF16 logits with a wider token-row pitch.
        beta=torch.randn(1, total_tokens, 40, device="cuda", dtype=torch.bfloat16)[
            :, :, 8 : 8 + num_heads
        ],
        A_log=(
            torch.randn(1, 1, num_heads, 1, device="cuda", dtype=torch.float32) * 0.2
        ).contiguous(),
        dt_bias=(
            torch.randn(num_heads * K, device="cuda", dtype=torch.float32) * 0.1
        ).contiguous(),
        state=(
            torch.randn(pool_size, num_heads, V, K, device="cuda", dtype=state_dtype)
            * 0.01
        ).contiguous(),
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def _extend(kernel, data, state, seq_lens, **kwargs):
    if getattr(kernel, "supports_cake_route_telemetry", False):
        kwargs["layer_id"] = 7
    beta = data["beta"]
    if isinstance(kernel, TritonKDAKernel):
        beta = torch.sigmoid(beta)
    return kernel.extend(
        data["q"].clone(),
        data["k"].clone(),
        data["v"].clone(),
        data["g"].clone(),
        beta,
        ssm_states=state,
        cache_indices=data["cache_indices"],
        query_start_loc=data["cu_seqlens"],
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_seq_lens_cpu=seq_lens,
        **kwargs,
    )


def _assert_close(name, actual, expected, atol=2e-2, rtol=2e-2):
    diff = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    assert bool((diff <= tol).all()), (
        f"{name}: max abs diff {diff.max().item():.4g}, "
        f"{int((diff > tol).sum())} elements beyond tolerance"
    )


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [
        (8, [128]),  # Kimi-Linear-48B TP4 local heads
        (16, [96]),  # TP2 local heads
        (16, [64, 160]),
        (8, [17, 64, 65, 127, 128, 255]),
        (16, [8192]),
        (8, [32768]),
    ],
)
def test_kda_prefill_prepared_export_matches_triton(num_heads, seq_lens):
    torch.manual_seed(num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(TritonKDAKernel(), data, state_triton, seq_lens)
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("prepared export must not fall back to Triton"),
    ):
        output_cake = _extend(CakeKDAKernel(), data, state_cake, seq_lens)
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
    )
    untouched = torch.ones(state_cake.shape[0], dtype=torch.bool, device="cuda")
    untouched[idx] = False
    assert torch.equal(state_cake[untouched], data["state"][untouched])


def test_kda_prefill_prepared_export_native_checkpoints():
    torch.manual_seed(1234)
    seq_lens = [200, 130]
    data = _make_inputs(seq_lens, 16)
    checkpoint_source = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    checkpoint_cu_starts = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int64)
    state_cake = data["state"].clone()
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("native checkpoints must not fall back to Triton"),
    ):
        output, h = _extend(
            CakeKDAKernel(),
            data,
            state_cake,
            seq_lens,
            return_intermediate_states=True,
            track_ssm_h_src=checkpoint_source,
            state_checkpoint_cu_starts=checkpoint_cu_starts,
            num_state_checkpoints=7,
            state_checkpoint_every_n_tokens=64,
        )
    assert output.shape == data["q"].shape
    assert h.shape == (1, 7, 16, V, K)
    assert h.dtype == state_cake.dtype
    assert torch.isfinite(h).all()
    # Every checkpoint is the state after 64*(i+1) tokens of its sequence: the
    # last checkpoint of a sequence whose length is a multiple of 64 matches the
    # final state; otherwise the final state extends beyond it. Check the
    # per-sequence 64-token prefix against an independent Triton run.
    for seq, (start, length) in enumerate(zip([0, 200], seq_lens)):
        prefix = 64
        sub = {
            key: (
                val[:, start : start + prefix]
                if key in ("q", "k", "v", "g", "beta")
                else val
            )
            for key, val in data.items()
        }
        sub["cu_seqlens"] = torch.tensor([0, prefix], device="cuda", dtype=torch.int32)
        sub["cache_indices"] = data["cache_indices"][seq : seq + 1]
        state_ref = data["state"].clone()
        _extend(TritonKDAKernel(), sub, state_ref, [prefix])
        first_checkpoint = int(checkpoint_cu_starts[seq])
        _assert_close(
            f"checkpoint[{seq}]",
            h[0, first_checkpoint],
            state_ref[int(sub["cache_indices"][0])],
            atol=1e-2,
            rtol=1e-2,
        )


def test_kda_prefill_policy_facade_keeps_bf16_pool_path():
    torch.manual_seed(7)
    data = _make_inputs([128], 8, state_dtype=torch.bfloat16)
    state = data["state"].clone()
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "auto"}):
        kernel = CakeKDAKernel()
        assert not kernel._cake_prefill_uses_prepared_export(state)
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "prepared"}):
        assert CakeKDAKernel()._cake_prefill_uses_prepared_export(state)
    with patch.dict(os.environ, {"SGLANG_KDA_CAKE_PREFILL_API": "facade"}):
        assert not CakeKDAKernel()._cake_prefill_uses_prepared_export(
            data["state"].float()
        )
