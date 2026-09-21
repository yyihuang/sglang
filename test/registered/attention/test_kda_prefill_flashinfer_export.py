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


def _extend(kernel, data, state, seq_lens, lower_bound=LOWER_BOUND, **kwargs):
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
        lower_bound=lower_bound,
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


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def test_kda_prefill_prepared_export_native_checkpoints():
    """Per-64-token-chunk states match Triton's intermediate states chunk by chunk.

    Checkpoint ``i`` of a sequence is the recurrent state at the start of its
    chunk ``i`` (checkpoint 0 is the initial state); the export stores them in
    BF16, so compare with a relative-L2 budget instead of elementwise atol.
    """
    torch.manual_seed(1234)
    seq_lens = [200, 130]
    data = _make_inputs(seq_lens, 16)
    checkpoint_source = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    checkpoint_cu_starts = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int64)
    cp_kwargs = dict(
        return_intermediate_states=True,
        track_ssm_h_src=checkpoint_source,
        state_checkpoint_cu_starts=checkpoint_cu_starts,
        num_state_checkpoints=7,
        state_checkpoint_every_n_tokens=64,
    )
    state_ref = data["state"].clone()
    output_ref, h_ref = _extend(
        TritonKDAKernel(), data, state_ref, seq_lens, **cp_kwargs
    )
    state_cake = data["state"].clone()
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("native checkpoints must not fall back to Triton"),
    ):
        output, h = _extend(CakeKDAKernel(), data, state_cake, seq_lens, **cp_kwargs)
    assert output.shape == data["q"].shape
    assert h.shape == h_ref.shape == (1, 7, 16, V, K)
    assert h.dtype == state_cake.dtype
    assert torch.isfinite(h).all()
    _assert_close("output", output, output_ref)
    idx = data["cache_indices"].long()
    assert _rel_l2(state_cake[idx], state_ref[idx]) < 1e-2
    starts = checkpoint_cu_starts.tolist()
    for seq in range(len(seq_lens)):
        first = starts[seq]
        # Checkpoint 0 of every sequence is its initial state, bit for bit.
        assert torch.equal(h[0, first], data["state"][idx[seq]])
        for j in range(first + 1, starts[seq + 1]):
            err = _rel_l2(h[0, j], h_ref[0, j])
            assert err < 1e-2, f"sequence {seq} chunk {j - first}: rel L2 {err:.4g}"


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [(16, [64, 160]), (8, [17, 64, 65, 127, 128, 255]), (16, [8192])],
)
def test_kda_prefill_prepared_export_unbounded_gate_matches_triton(num_heads, seq_lens):
    """Kimi-Linear has no gate lower bound; the export serves the unbounded softplus gate."""
    torch.manual_seed(99 + num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    state_triton = data["state"].clone()
    state_cake = data["state"].clone()
    output_triton = _extend(
        TritonKDAKernel(), data, state_triton, seq_lens, lower_bound=None
    )
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("unbounded gate must not fall back to Triton"),
    ):
        output_cake = _extend(
            CakeKDAKernel(), data, state_cake, seq_lens, lower_bound=None
        )
    _assert_close("output", output_cake, output_triton)
    idx = data["cache_indices"].long()
    _assert_close(
        "final_state", state_cake[idx], state_triton[idx], atol=1e-2, rtol=1e-2
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
