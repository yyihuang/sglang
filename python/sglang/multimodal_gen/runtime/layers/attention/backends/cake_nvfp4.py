# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class _CakeNVFP4CorrectionWorkspace(NamedTuple):
    q_padded: torch.Tensor
    q_mean: torch.Tensor
    q_mean_fp32: torch.Tensor
    k_mean: torch.Tensor
    k_centered: torch.Tensor
    k_centered_fp32_t: torch.Tensor
    qk_correction: torch.Tensor


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
        self._workspace_key = None
        self._workspace = None
        self._output = None
        self._correction_workspace = None

    @staticmethod
    def _allocate_correction_workspace(
        query: torch.Tensor,
    ) -> _CakeNVFP4CorrectionWorkspace:
        batch, seq_len, heads, head_dim = query.shape
        padded_seq_len = (seq_len + 511) // 512 * 512
        q_blocks = padded_seq_len // 128
        common = {"device": query.device, "dtype": query.dtype}
        return _CakeNVFP4CorrectionWorkspace(
            q_padded=torch.empty((batch, padded_seq_len, heads, head_dim), **common),
            q_mean=torch.empty((batch, q_blocks, heads, head_dim), **common),
            q_mean_fp32=torch.empty(
                (batch, heads, q_blocks, head_dim),
                device=query.device,
                dtype=torch.float32,
            ),
            k_mean=torch.empty((batch, 1, heads, head_dim), **common),
            k_centered=torch.empty_like(query),
            k_centered_fp32_t=torch.empty(
                (batch, heads, head_dim, padded_seq_len),
                device=query.device,
                dtype=torch.float32,
            ),
            qk_correction=torch.empty(
                (batch, heads, q_blocks, padded_seq_len),
                device=query.device,
                dtype=torch.float32,
            ),
        )

    def _get_workspace_and_output(self, query: torch.Tensor):
        try:
            from flashinfer import allocate_cake_nvfp4_attention_workspace
        except ImportError as error:
            raise ImportError(
                "Cake NVFP4 attention requires a FlashInfer build that exports "
                "allocate_cake_nvfp4_attention_workspace."
            ) from error

        key = (
            tuple(query.shape),
            query.device.type,
            query.device.index,
            query.dtype,
        )
        if key != self._workspace_key:
            self._workspace = allocate_cake_nvfp4_attention_workspace(
                query, qkv_layout="NHD"
            )
            self._output = torch.empty_like(query, dtype=torch.bfloat16)
            self._correction_workspace = self._allocate_correction_workspace(query)
            self._workspace_key = key
        return self._workspace, self._output, self._correction_workspace

    @staticmethod
    def _prepare_qk_correction(
        query: torch.Tensor,
        key: torch.Tensor,
        workspace: _CakeNVFP4CorrectionWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, heads, head_dim = query.shape
        padded_seq_len = workspace.q_padded.shape[1]
        q_blocks = padded_seq_len // 128

        workspace.q_padded.zero_()
        workspace.q_padded[:, :seq_len].copy_(query)
        q_grouped = workspace.q_padded.view(batch, q_blocks, 128, heads, head_dim)
        torch.mean(q_grouped, dim=2, out=workspace.q_mean)
        torch.sub(q_grouped, workspace.q_mean.unsqueeze(2), out=q_grouped)

        torch.mean(key, dim=1, keepdim=True, out=workspace.k_mean)
        torch.sub(key, workspace.k_mean, out=workspace.k_centered)
        workspace.q_mean_fp32.copy_(workspace.q_mean.permute(0, 2, 1, 3))
        workspace.k_centered_fp32_t.zero_()
        workspace.k_centered_fp32_t[..., :seq_len].copy_(
            workspace.k_centered.permute(0, 2, 3, 1)
        )
        torch.matmul(
            workspace.q_mean_fp32,
            workspace.k_centered_fp32_t,
            out=workspace.qk_correction,
        )

        centered_query = workspace.q_padded[:, :seq_len]
        correction = workspace.qk_correction.view(
            batch * heads, q_blocks, padded_seq_len
        )
        return centered_query, workspace.k_centered, correction

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
        if query.shape[0] != 1:
            raise ValueError(
                "Cake NVFP4 Wan serving is qualified only for batch=1, "
                f"got batch={query.shape[0]}"
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
        if (
            not query.is_contiguous()
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError("Cake NVFP4 attention requires contiguous NHD Q/K/V")

        try:
            from flashinfer import nvfp4_attention
        except ImportError as error:
            raise ImportError(
                "Cake NVFP4 attention requires a FlashInfer build that exports "
                "flashinfer.nvfp4_attention."
            ) from error

        workspace, output, correction_workspace = self._get_workspace_and_output(query)
        centered_query, centered_key, qk_correction = self._prepare_qk_correction(
            query, key, correction_workspace
        )
        return nvfp4_attention(
            centered_query,
            centered_key,
            value,
            sm_scale=self.softmax_scale,
            causal=False,
            backend="cake",
            qkv_layout="NHD",
            workspace=workspace,
            qk_correction=qk_correction,
            out=output,
        )
