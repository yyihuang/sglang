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
    num_queries: int,
    swa_width: int,
    compressed_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return legal row IDs for two per-query gathered scratch pools."""

    swa = torch.arange(num_queries * swa_width, dtype=torch.int32, device=device).view(
        num_queries, swa_width
    )
    if compressed_width == 0:
        return swa
    compressed = torch.arange(
        num_queries * compressed_width, dtype=torch.int32, device=device
    ).view(num_queries, compressed_width)
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
            active_lens = swa_active_lens.reshape(-1).to(
                torch.int32
            ) + compressed_active_lens.reshape(-1).to(torch.int32)

        expected_width = 128 + compressed_width
        if self._indices is None or self._indices.shape != (
            num_queries,
            expected_width,
        ):
            self._indices = _rebased_indices(
                num_queries,
                128,
                compressed_width,
                device=self.device,
            )
        workspace = self._reduction_workspace(
            num_queries=num_queries,
            num_heads=num_heads,
            sparse_width=expected_width,
        )
        out = self._output(q.shape)

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
        if self.launch_count == 1:
            logger.info(
                "SGLang DSV4 decode routed to FlashInfer backend='cake' "
                "without fallback (packed-KV gather width=%d)",
                expected_width,
            )
        return result


__all__ = [
    "CakeDsv4DecodeWorkspace",
    "is_cake_dsv4_enabled",
]
