"""GB300 correctness/performance probe for SGLang's strict DSV4 CAKE adapter.

This test uses SGLang's real 584-byte packed KV layout and compares the full
adapter call (selected-row dequantization plus CAKE attention) against the
production ``flash_mla_with_kvcache`` call. Reportable timings are CUPTI GPU
activity unions, so launch gaps are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass

import torch
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import _set_k_and_s_triton
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.dsv4.cake_backend import CakeDsv4DecodeWorkspace

HEAD_DIM = 512
LOCAL_HEADS = 32
KERNEL_HEADS = 64


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    compressed_width: int
    compressed_page_size: int | None
    swa_active: int = 128
    compressed_active: int | None = None
    max_seq_len: int = 4096


CASES = (
    Case("tp4_b1_swa128", 1, 0, None),
    Case("tp4_b8_swa128", 8, 0, None),
    Case("tp4_b1_c4_topk512", 1, 512, 64),
    Case("tp4_b8_c4_topk512", 8, 512, 64),
    Case("tp4_b16_c4_topk512", 16, 512, 64),
    Case("tp4_b1_topk4x", 1, 1024, 64),
    Case("tp4_b8_topk4x", 8, 1024, 64),
    Case("tp4_b1_topk128x", 1, 128, 2),
    Case("tp4_b8_topk128x", 8, 128, 2),
    Case(
        "tp4_b1_c128_capacity8256_live1",
        1,
        8256,
        2,
        swa_active=7,
        compressed_active=1,
        max_seq_len=7,
    ),
)


def _packed_cache(rows: int, page_size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = (
        torch.randn((rows, HEAD_DIM), generator=generator, device="cuda") * 0.05
    ).to(torch.bfloat16)
    packed = quant_to_nope_fp8_rope_bf16_pack_triton(values)
    # Keep at least two pages so PyTorch preserves the padded outer stride in
    # the 4-D FlashMLA view.  A degenerate one-page allocation is allowed to
    # collapse that stride and does not match SGLang's real multi-page pool.
    num_pages = max(2, math.ceil(rows / page_size))
    raw_page_bytes = page_size * 584
    padded_page_bytes = math.ceil(raw_page_bytes / 576) * 576
    cache = torch.zeros(
        (num_pages, padded_page_bytes), dtype=torch.uint8, device="cuda"
    )
    _set_k_and_s_triton(
        cache,
        torch.arange(rows, dtype=torch.int32, device="cuda"),
        packed,
        page_size,
    )
    return cache


def _dense_reference(
    *,
    q: torch.Tensor,
    swa: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_active_lens: torch.Tensor,
    compressed: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_active_lens: torch.Tensor | None,
    softmax_scale: float,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact dense-BF16 semantics consumed by CAKE."""

    batch, _, heads, _ = q.shape
    swa_width = swa_indices.shape[-1]
    swa_rows = swa.reshape(batch, swa_width, HEAD_DIM).float()
    compressed_width = 0 if compressed_indices is None else compressed_indices.shape[-1]
    compressed_rows = (
        None
        if compressed is None
        else compressed.reshape(batch, compressed_width, HEAD_DIM).float()
    )
    output = torch.zeros((batch, heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    for row in range(batch):
        gathered = swa_rows[row]
        valid = swa_indices[row].reshape(-1) >= 0
        active_len = int(swa_active_lens[row])
        if compressed_rows is not None:
            assert compressed_indices is not None
            assert compressed_active_lens is not None
            gathered = torch.cat((gathered, compressed_rows[row]), dim=0)
            valid = torch.cat(
                (valid, compressed_indices[row].reshape(-1) >= 0), dim=0
            )
            # CAKE reserves the first 128 physical positions for SWA.
            active_len = 128 + int(compressed_active_lens[row])
        valid &= torch.arange(valid.numel(), device=q.device) < active_len
        scores = q[row, 0].float() @ gathered[valid].transpose(0, 1)
        scores *= softmax_scale
        sink = sinks.reshape(-1, 1)
        row_max = torch.maximum(scores.amax(-1, keepdim=True), sink)
        numerator = torch.exp(scores - row_max)
        probabilities = numerator / (
            numerator.sum(-1, keepdim=True) + torch.exp(sink - row_max)
        )
        output[row] = probabilities @ gathered[valid]
    return output.to(torch.bfloat16).unsqueeze(1)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f = actual.float()
    expected_f = expected.float()
    close = torch.isclose(actual_f, expected_f, atol=1e-2, rtol=1e-2)
    return {
        "strict_close": bool(close.all()),
        "outside_tolerance": int((~close).sum()),
        "elements": close.numel(),
        "max_abs_err": float((actual_f - expected_f).abs().max()),
    }


def _run_case(
    case: Case,
    *,
    benchmark: bool,
    diagnostic: bool,
    adapter: CakeDsv4DecodeWorkspace,
) -> dict:
    torch.manual_seed(1000 + case.batch + case.compressed_width)
    # TP4 owns 32 of the model's 128 heads, but the production DSV4 path pads
    # to FlashMLA's H64 specialization and discards the upper 32 outputs.
    q = torch.zeros(
        (case.batch, 1, KERNEL_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q[:, :, :LOCAL_HEADS] = (
        torch.randn((case.batch, 1, LOCAL_HEADS, HEAD_DIM), device="cuda") * 0.05
    ).to(torch.bfloat16)
    sinks = torch.zeros(KERNEL_HEADS, dtype=torch.float32, device="cuda")
    sinks[:LOCAL_HEADS] = (
        torch.randn(LOCAL_HEADS, dtype=torch.float32, device="cuda") * 0.05
    )
    scale = HEAD_DIM**-0.5

    swa_width = 128
    swa_rows = case.batch * swa_width
    swa_cache = _packed_cache(swa_rows, 128, seed=2000 + case.batch)
    swa_indices = torch.arange(swa_rows, dtype=torch.int32, device="cuda").view(
        case.batch, 1, swa_width
    )
    swa_indices[:, :, case.swa_active :] = -1
    swa_lens = torch.full(
        (case.batch,), case.swa_active, dtype=torch.int32, device="cuda"
    )

    if case.compressed_width:
        assert case.compressed_page_size is not None
        compressed_rows = case.batch * case.compressed_width
        compressed_cache = _packed_cache(
            compressed_rows,
            case.compressed_page_size,
            seed=3000 + case.compressed_width + case.batch,
        )
        compressed_indices = torch.arange(
            compressed_rows, dtype=torch.int32, device="cuda"
        ).view(case.batch, 1, case.compressed_width)
        compressed_active = (
            case.compressed_width
            if case.compressed_active is None
            else case.compressed_active
        )
        compressed_indices[:, :, compressed_active:] = -1
        compressed_lens = torch.full(
            (case.batch,), compressed_active, dtype=torch.int32, device="cuda"
        )
    else:
        compressed_cache = None
        compressed_indices = None
        compressed_lens = None

    scheduler_metadata = get_mla_metadata()[0]
    packed_swa_4d = swa_cache[:, : 128 * 584].view(-1, 128, 1, 584)
    packed_compressed_4d = None
    if compressed_cache is not None:
        packed_compressed_4d = compressed_cache[
            :, : case.compressed_page_size * 584
        ].view(-1, case.compressed_page_size, 1, 584)

    def baseline():
        return flash_mla_with_kvcache(
            q=q,
            k_cache=packed_swa_4d,
            head_dim_v=HEAD_DIM,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=scheduler_metadata,
            softmax_scale=scale,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            attn_sink=sinks,
            extra_k_cache=packed_compressed_4d,
            extra_indices_in_kvcache=compressed_indices,
            extra_topk_length=compressed_lens,
        )[0]

    def candidate():
        return adapter.run(
            q=q,
            packed_swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_active_lens=swa_lens,
            swa_page_size=128,
            packed_compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_active_lens=compressed_lens,
            compressed_page_size=case.compressed_page_size,
            seq_lens=torch.full(
                (case.batch,), case.max_seq_len, dtype=torch.int32, device="cuda"
            ),
            max_seq_len=case.max_seq_len,
            softmax_scale=scale,
            sinks=sinks,
        )

    expected = baseline()
    actual = candidate()
    torch.cuda.synchronize()
    actual_local = actual[:, :, :LOCAL_HEADS]
    expected_local = expected[:, :, :LOCAL_HEADS]
    packed_stats = _error_stats(actual_local, expected_local)
    if not diagnostic:
        torch.testing.assert_close(actual_local, expected_local, atol=1e-2, rtol=1e-2)
    result = {
        "case": case.name,
        "correct": packed_stats["strict_close"],
        "kernel_heads": KERNEL_HEADS,
        "model_local_heads": LOCAL_HEADS,
        "max_abs_err": packed_stats["max_abs_err"],
        "cake_vs_packed": packed_stats,
        "cake_route_count": adapter.launch_count,
    }
    if diagnostic:
        swa_dense = adapter._swa[: case.batch * swa_width]
        compressed_dense = None
        if compressed_indices is not None:
            compressed_dense = adapter._compressed[
                : case.batch * compressed_indices.shape[-1]
            ]
        dense = _dense_reference(
            q=q,
            swa=swa_dense,
            swa_indices=swa_indices,
            swa_active_lens=swa_lens,
            compressed=compressed_dense,
            compressed_indices=compressed_indices,
            compressed_active_lens=compressed_lens,
            softmax_scale=scale,
            sinks=sinks,
        )[:, :, :LOCAL_HEADS]
        result["cake_vs_dense_reference"] = _error_stats(actual_local, dense)
        result["packed_vs_dense_reference"] = _error_stats(expected_local, dense)
    if benchmark:
        from loom.bench import bench_gpu_time

        timings = {}
        for repeat in range(3):
            order = (("baseline", baseline), ("cake", candidate))
            if repeat % 2:
                order = tuple(reversed(order))
            for name, fn in order:
                timing = bench_gpu_time(
                    fn,
                    warmup_ms=20,
                    bench_ms=100,
                    cold_l2=True,
                    use_cupti=True,
                )
                timings.setdefault(name, []).extend(timing.active_union_times_ms)
        baseline_ms = statistics.median(timings["baseline"])
        cake_ms = statistics.median(timings["cake"])
        result.update(
            timing_metric="cupti_active_union_ms",
            baseline_active_union_ms=baseline_ms,
            cake_active_union_ms=cake_ms,
            speedup=baseline_ms / cake_ms,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(
            f"expected GB300 SM103, got {torch.cuda.get_device_name()} "
            f"SM{torch.cuda.get_device_capability()}"
        )
    # Production owns one workspace across all layers.  Reuse it while widths
    # and batch sizes change so the probe covers scratch growth and c4/c128/SWA
    # alternation, not only isolated launches with fresh allocations.
    adapter = CakeDsv4DecodeWorkspace(torch.device("cuda"))
    selected = CASES
    if args.cases:
        requested = set(args.cases)
        selected = tuple(case for case in CASES if case.name in requested)
        missing = requested - {case.name for case in selected}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")
    rows = [
        _run_case(
            case,
            benchmark=args.benchmark,
            diagnostic=args.diagnostic,
            adapter=adapter,
        )
        for case in selected
    ]
    print("SGLANG_DSV4_CAKE_GPU_ROWS=" + json.dumps(rows, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
