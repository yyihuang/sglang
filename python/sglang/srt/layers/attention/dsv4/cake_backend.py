"""Explicit CAKE adapter for DeepSeek-V4 decode on NVIDIA GB300.

SGLang stores DeepSeek-V4 KV rows in a mixed packed representation (448 FP8
NOPE values, 64 BF16 RoPE values, and seven block scales).  FlashInfer's CAKE
backend consumes separate dense BF16 SWA and compressed pools.  This adapter
dequantizes only the rows selected by the current decode step, rebases their
indices into reusable dense scratch pools, and calls ``backend="cake"``.

The adapter intentionally has no fallback.  It is an opt-in validation path
selected with ``SGLANG_HACK_FLASHMLA_BACKEND=cake`` on SM103.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_HEAD_DIM = 512
_TILE_KV = 128


def is_cake_dsv4_enabled() -> bool:
    return is_cuda() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "cake"


def _flashinfer_dsv4():
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    return trtllm_batch_decode_sparse_mla_dsv4


def _rebased_indices(
    swa_indices: torch.Tensor,
    compressed_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Rebase valid source rows while preserving padded ``-1`` sentinels."""

    num_queries, swa_width = swa_indices.shape
    device = swa_indices.device
    swa = torch.arange(num_queries * swa_width, dtype=torch.int32, device=device).view(
        num_queries, swa_width
    )
    swa = swa.masked_fill(swa_indices < 0, -1)
    if compressed_indices is None:
        return swa
    compressed_width = compressed_indices.shape[1]
    compressed = torch.arange(
        num_queries * compressed_width, dtype=torch.int32, device=device
    ).view(num_queries, compressed_width)
    compressed = compressed.masked_fill(compressed_indices < 0, -1)
    return torch.cat((swa, compressed), dim=-1)


class CakeDsv4DecodeWorkspace:
    """Reusable scratch and strict CAKE launch path for ordinary decode."""

    def __init__(self, device: torch.device):
        self.device = device
        self._swa: torch.Tensor | None = None
        self._compressed: torch.Tensor | None = None
        self._indices: torch.Tensor | None = None
        self._workspace: torch.Tensor | None = None
        self._out: torch.Tensor | None = None
        self._logged_shapes: set[tuple[int, int, int]] = set()
        self.launch_count = 0

    def _bf16_rows(self, name: str, rows: int) -> torch.Tensor:
        current = getattr(self, name)
        if current is None or current.shape[0] < rows:
            current = torch.empty(
                (rows, 1, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=self.device,
            )
            setattr(self, name, current)
        return current[:rows]

    def _output(self, shape: torch.Size) -> torch.Tensor:
        if self._out is None or any(
            current < required
            for current, required in zip(self._out.shape, shape, strict=True)
        ):
            self._out = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        return self._out[: shape[0], : shape[1], : shape[2], : shape[3]]

    def _reduction_workspace(
        self, *, num_queries: int, num_heads: int, sparse_width: int
    ) -> torch.Tensor:
        num_splits = (sparse_width + _TILE_KV - 1) // _TILE_KV
        lse_elems = num_queries * num_heads * num_splits
        partial_o_bytes = 0
        if num_splits > 1:
            partial_o_bytes = lse_elems * _HEAD_DIM * torch.bfloat16.itemsize
        lse_offset = (partial_o_bytes + 15) & ~15
        required = lse_offset + lse_elems * torch.float32.itemsize
        if self._workspace is None or self._workspace.numel() < required:
            self._workspace = torch.empty(
                required, dtype=torch.uint8, device=self.device
            )
        return self._workspace[:required]

    def run(
        self,
        *,
        q: torch.Tensor,
        packed_swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_active_lens: torch.Tensor,
        swa_page_size: int,
        packed_compressed_cache: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
        compressed_active_lens: torch.Tensor | None,
        compressed_page_size: int | None,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        softmax_scale: float,
        sinks: torch.Tensor,
    ) -> torch.Tensor:
        if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != _HEAD_DIM:
            raise ValueError(
                "CAKE DSV4 SGLang adapter requires decode Q [B, 1, H, 512], "
                f"got {tuple(q.shape)}"
            )
        num_queries, _, num_heads, _ = q.shape
        swa_indices = swa_indices.reshape(num_queries, -1)
        if swa_indices.shape[1] != 128:
            raise ValueError(
                "CAKE DSV4 SGLang adapter requires exactly 128 SWA slots, "
                f"got {swa_indices.shape[1]}"
            )

        safe_swa_indices = swa_indices.reshape(-1).clamp_min(0).contiguous()
        swa = self._bf16_rows("_swa", safe_swa_indices.numel())
        dequantize_k_cache_paged(
            packed_swa_cache,
            safe_swa_indices,
            page_size=swa_page_size,
            out=swa,
        )

        compressed_width = 0
        if compressed_indices is None:
            compressed = self._bf16_rows("_compressed", 1)
            active_lens = swa_active_lens.reshape(-1).to(torch.int32)
        else:
            if (
                packed_compressed_cache is None
                or compressed_active_lens is None
                or compressed_page_size is None
            ):
                raise ValueError(
                    "compressed CAKE DSV4 decode requires cache, active lengths, "
                    "and page size"
                )
            compressed_indices = compressed_indices.reshape(num_queries, -1)
            if compressed_page_size == 2:
                # c128 metadata is sized for the model's maximum context.  Only
                # the current batch's live c128 prefix may enter the CAKE launch;
                # otherwise a short decode would spuriously launch thousands of
                # empty split CTAs and exceed the reducer's supported split set.
                live_c128_width = max(1, (max_seq_len + 127) // 128)
                if live_c128_width > compressed_indices.shape[1]:
                    raise ValueError(
                        "live c128 extent exceeds metadata capacity: "
                        f"extent={live_c128_width}, "
                        f"capacity={compressed_indices.shape[1]}"
                    )
                compressed_indices = compressed_indices[:, :live_c128_width]
            compressed_width = compressed_indices.shape[1]
            safe_compressed_indices = (
                compressed_indices.reshape(-1).clamp_min(0).contiguous()
            )
            compressed = self._bf16_rows("_compressed", safe_compressed_indices.numel())
            dequantize_k_cache_paged(
                packed_compressed_cache,
                safe_compressed_indices,
                page_size=compressed_page_size,
                out=compressed,
            )
            # CAKE's split 0 is always the 128-column SWA region; compressed
            # rows begin at split 1.  Keep that physical split boundary even
            # when the live SWA prefix is shorter than 128.  The preserved -1
            # sentinels above mask the unused tail of split 0.
            active_lens = 128 + compressed_active_lens.reshape(-1).to(torch.int32)

        expected_width = 128 + compressed_width
        self._indices = _rebased_indices(
            swa_indices,
            compressed_indices,
        )
        workspace = self._reduction_workspace(
            num_queries=num_queries,
            num_heads=num_heads,
            sparse_width=expected_width,
        )
        out = self._output(q.shape)

        if os.getenv("SGLANG_CAKE_DSV4_DEBUG_SHAPES") == "1":
            lens_host = active_lens.tolist()
            if any(length < 0 or length > expected_width for length in lens_host):
                raise ValueError(
                    "CAKE DSV4 active length exceeds gathered sparse width: "
                    f"lens={lens_host}, width={expected_width}"
                )
            logger.info(
                "CAKE DSV4 debug launch: queries=%d heads=%d width=%d "
                "active_lens=%s workspace_bytes=%d",
                num_queries,
                num_heads,
                expected_width,
                lens_host,
                workspace.numel(),
            )

        result = _flashinfer_dsv4()(
            q,
            swa,
            workspace,
            sparse_indices=self._indices,
            compressed_kv_cache=compressed,
            sparse_topk_lens=active_lens,
            seq_lens=seq_lens[:num_queries].to(torch.int32),
            out=out,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            sinks=sinks,
            # Gathered scratch is a flat 3-D [row, head, dim] pool.  FlashInfer
            # represents that form as HND; NHD is reserved for paged 4-D input.
            kv_layout="HND",
            enable_pdl=False,
            backend="cake",
        )
        self.launch_count += 1
        route_shape = (num_queries, num_heads, expected_width)
        if route_shape not in self._logged_shapes:
            logger.info(
                "SGLang DSV4 decode routed to FlashInfer backend='cake' "
                "without fallback (queries=%d, heads=%d, packed-KV gather "
                "width=%d, compressed-page-size=%s)",
                num_queries,
                num_heads,
                expected_width,
                compressed_page_size,
            )
            self._logged_shapes.add(route_shape)
        return result


__all__ = [
    "CakeDsv4DecodeWorkspace",
    "is_cake_dsv4_enabled",
]
