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
from loom.bench import bench_gpu_time
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import _set_k_and_s_triton
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.dsv4.cake_backend import CakeDsv4DecodeWorkspace

HEAD_DIM = 512
HEADS = 32


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    compressed_width: int
    compressed_page_size: int | None


CASES = (
    Case("tp4_b1_swa128", 1, 0, None),
    Case("tp4_b8_swa128", 8, 0, None),
    Case("tp4_b1_topk4x", 1, 1024, 64),
    Case("tp4_b8_topk4x", 8, 1024, 64),
    Case("tp4_b1_topk128x", 1, 128, 2),
    Case("tp4_b8_topk128x", 8, 128, 2),
)


def _packed_cache(rows: int, page_size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = (
        torch.randn((rows, HEAD_DIM), generator=generator, device="cuda") * 0.05
    ).to(torch.bfloat16)
    packed = quant_to_nope_fp8_rope_bf16_pack_triton(values)
    num_pages = math.ceil(rows / page_size)
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


def _run_case(case: Case, *, benchmark: bool) -> dict:
    torch.manual_seed(1000 + case.batch + case.compressed_width)
    q = (torch.randn((case.batch, 1, HEADS, HEAD_DIM), device="cuda") * 0.05).to(
        torch.bfloat16
    )
    sinks = torch.randn(HEADS, dtype=torch.float32, device="cuda") * 0.05
    scale = HEAD_DIM**-0.5

    swa_width = 128
    swa_rows = case.batch * swa_width
    swa_cache = _packed_cache(swa_rows, 128, seed=2000 + case.batch)
    swa_indices = torch.arange(swa_rows, dtype=torch.int32, device="cuda").view(
        case.batch, 1, swa_width
    )
    swa_lens = torch.full((case.batch,), swa_width, dtype=torch.int32, device="cuda")

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
        compressed_lens = torch.full(
            (case.batch,),
            case.compressed_width,
            dtype=torch.int32,
            device="cuda",
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

    adapter = CakeDsv4DecodeWorkspace(torch.device("cuda"))

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
            seq_lens=torch.full((case.batch,), 4096, dtype=torch.int32, device="cuda"),
            softmax_scale=scale,
            sinks=sinks,
        )

    expected = baseline()
    actual = candidate()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    result = {
        "case": case.name,
        "correct": True,
        "max_abs_err": float((actual.float() - expected.float()).abs().max()),
        "cake_route_count": adapter.launch_count,
    }
    if benchmark:
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
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(
            f"expected GB300 SM103, got {torch.cuda.get_device_name()} "
            f"SM{torch.cuda.get_device_capability()}"
        )
    rows = [_run_case(case, benchmark=args.benchmark) for case in CASES]
    print("SGLANG_DSV4_CAKE_GPU_ROWS=" + json.dumps(rows, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
