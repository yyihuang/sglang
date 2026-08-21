# SPDX-License-Identifier: Apache-2.0

import threading
import weakref
from dataclasses import dataclass

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class _WanHybridSharedScratch:
    workspace: object


_SHARED_SCRATCH_LOCK = threading.Lock()
_SHARED_SCRATCH: weakref.WeakValueDictionary[tuple, _WanHybridSharedScratch] = (
    weakref.WeakValueDictionary()
)
_HIT_COUNT_LOCK = threading.Lock()
_SUCCESSFUL_FORWARD_HIT_COUNT = 0


def reset_wan_hybrid_hit_count() -> None:
    global _SUCCESSFUL_FORWARD_HIT_COUNT
    with _HIT_COUNT_LOCK:
        _SUCCESSFUL_FORWARD_HIT_COUNT = 0


def read_wan_hybrid_hit_count() -> int:
    with _HIT_COUNT_LOCK:
        return _SUCCESSFUL_FORWARD_HIT_COUNT


def _record_successful_wan_hybrid_forward(result: torch.Tensor) -> torch.Tensor:
    global _SUCCESSFUL_FORWARD_HIT_COUNT
    with _HIT_COUNT_LOCK:
        _SUCCESSFUL_FORWARD_HIT_COUNT += 1
    return result


class WanHybridAttentionBackend(AttentionBackend):
    """Exact-shape Wan self-attention through FlashInfer's public API."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.WAN_HYBRID

    @staticmethod
    def get_impl_cls() -> type["WanHybridAttentionImpl"]:
        return WanHybridAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class WanHybridAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del prefix, extra_impl_args
        if head_size != 128:
            raise ValueError(
                f"Wan hybrid attention requires head_size=128, got {head_size}"
            )
        if causal:
            raise ValueError("Wan hybrid serving supports noncausal attention only")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads != num_heads:
            raise ValueError(
                "Wan hybrid attention requires equal query and KV head counts, "
                f"got {num_heads} and {num_kv_heads}"
            )
        if num_heads != 40:
            raise ValueError(
                f"Wan hybrid Wan serving requires num_heads=40, got {num_heads}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self._workspace_key = None
        self._shared_scratch = None
        self._output = None

    def _get_workspace_and_output(self, query: torch.Tensor):
        try:
            from flashinfer import WanHybridAttentionWorkspace
        except ImportError as error:
            raise ImportError(
                "Wan hybrid attention requires a FlashInfer build that exports "
                "WanHybridAttentionWorkspace."
            ) from error

        key = (
            tuple(query.shape),
            query.device.type,
            query.device.index,
            query.dtype,
            torch.cuda.current_stream(query.device).cuda_stream,
        )
        if key != self._workspace_key:
            with _SHARED_SCRATCH_LOCK:
                shared_scratch = _SHARED_SCRATCH.get(key)
                if shared_scratch is None:
                    shared_scratch = _WanHybridSharedScratch(
                        workspace=WanHybridAttentionWorkspace(query.device)
                    )
                    _SHARED_SCRATCH[key] = shared_scratch
            self._shared_scratch = shared_scratch
            self._output = torch.empty_like(query, dtype=torch.bfloat16)
            self._workspace_key = key
        return self._shared_scratch.workspace, self._output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        del attn_metadata
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "Wan hybrid attention supports dense self-attention only; "
                f"got q={tuple(query.shape)}, k={tuple(key.shape)}, "
                f"v={tuple(value.shape)}"
            )
        if query.ndim != 4 or query.shape[2:] != (
            self.num_heads,
            self.head_size,
        ):
            raise ValueError(
                "Wan hybrid attention expects [batch, seq_len, heads, 128], "
                f"got {tuple(query.shape)}"
            )
        if query.shape[0] != 1:
            raise ValueError(
                "Wan hybrid Wan serving is qualified only for batch=1, "
                f"got batch={query.shape[0]}"
            )
        if query.shape[1] != 4800:
            raise ValueError(
                "Wan hybrid Wan serving is qualified only for sequence length "
                f"4800, got {query.shape[1]}"
            )
        if query.device.type != "cuda":
            raise ValueError("Wan hybrid attention requires CUDA tensors")
        if query.dtype != torch.bfloat16:
            raise ValueError(
                "Wan hybrid serving requires BF16 Q/K/V because its output is BF16, "
                f"got {query.dtype}"
            )
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("Wan hybrid attention requires matching Q/K/V dtypes")
        if (
            not query.is_contiguous()
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError("Wan hybrid attention requires contiguous NHD Q/K/V")

        try:
            from flashinfer import (
                is_wan_hybrid_attention_available,
                wan_hybrid_attention,
            )
        except ImportError as error:
            raise ImportError(
                "Wan hybrid attention requires a FlashInfer build that exports "
                "the public wan_hybrid attention API."
            ) from error

        if not is_wan_hybrid_attention_available(query.device):
            raise NotImplementedError(
                "FlashInfer wan_hybrid attention is unavailable on this device"
            )

        workspace, output = self._get_workspace_and_output(query)
        return _record_successful_wan_hybrid_forward(
            wan_hybrid_attention(
                query,
                key,
                value,
                out=output,
                workspace=workspace,
                sm_scale=self.softmax_scale,
                qkv_layout="NHD",
                causal=False,
            )
        )
