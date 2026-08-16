# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class CakeNVFP4AttentionBackend(AttentionBackend):
    """Dense Wan self-attention through FlashInfer's Cake NVFP4 backend."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.CAKE_NVFP4

    @staticmethod
    def get_impl_cls() -> type["CakeNVFP4AttentionImpl"]:
        return CakeNVFP4AttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class CakeNVFP4AttentionImpl(AttentionImpl):
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
                f"Cake NVFP4 attention requires head_size=128, got {head_size}"
            )
        if causal:
            raise ValueError("Cake NVFP4 serving supports noncausal attention only")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads != num_heads:
            raise ValueError(
                "Cake NVFP4 attention requires equal query and KV head counts, "
                f"got {num_heads} and {num_kv_heads}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale

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
                "Cake NVFP4 attention supports dense self-attention only; "
                f"got q={tuple(query.shape)}, k={tuple(key.shape)}, "
                f"v={tuple(value.shape)}"
            )
        if query.ndim != 4 or query.shape[2:] != (
            self.num_heads,
            self.head_size,
        ):
            raise ValueError(
                "Cake NVFP4 attention expects [batch, seq_len, heads, 128], "
                f"got {tuple(query.shape)}"
            )
        if query.device.type != "cuda":
            raise ValueError("Cake NVFP4 attention requires CUDA tensors")
        if query.dtype != torch.bfloat16:
            raise ValueError(
                "Cake NVFP4 serving requires BF16 Q/K/V because its output is BF16, "
                f"got {query.dtype}"
            )
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("Cake NVFP4 attention requires matching Q/K/V dtypes")

        try:
            from flashinfer import nvfp4_attention
        except ImportError as error:
            raise ImportError(
                "Cake NVFP4 attention requires a FlashInfer build that exports "
                "flashinfer.nvfp4_attention."
            ) from error

        query_hnd = query.transpose(1, 2).contiguous()
        # Translating every key by the same per-head vector adds a constant to
        # each row of QK^T, so softmax attention is unchanged. Centering K
        # before FP4 quantization removes the common-mode component without
        # requiring a correction term in the attention kernel.
        key_centered = key - key.mean(dim=1, keepdim=True)
        key_hnd = key_centered.transpose(1, 2).contiguous()
        value_hnd = value.transpose(1, 2).contiguous()
        output_hnd = nvfp4_attention(
            query_hnd,
            key_hnd,
            value_hnd,
            sm_scale=self.softmax_scale,
            causal=False,
            backend="cake",
        )
        return output_hnd.transpose(1, 2).contiguous()
