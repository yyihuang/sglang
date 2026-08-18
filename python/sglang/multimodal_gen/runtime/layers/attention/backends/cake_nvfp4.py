# SPDX-License-Identifier: Apache-2.0

import threading
import weakref
from dataclasses import dataclass
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
    q_mean: torch.Tensor
    k_mean: torch.Tensor
    qk_correction: torch.Tensor


@dataclass
class _CakeNVFP4SharedScratch:
    packed: object
    correction: _CakeNVFP4CorrectionWorkspace


_SHARED_SCRATCH_LOCK = threading.Lock()
_SHARED_SCRATCH: weakref.WeakValueDictionary[tuple, _CakeNVFP4SharedScratch] = (
    weakref.WeakValueDictionary()
)


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
        self._shared_scratch = None
        self._output = None

    @staticmethod
    def _allocate_correction_workspace(
        query: torch.Tensor,
    ) -> _CakeNVFP4CorrectionWorkspace:
        batch, seq_len, heads, head_dim = query.shape
        padded_seq_len = (seq_len + 511) // 512 * 512
        q_blocks = padded_seq_len // 128
        common = {"device": query.device, "dtype": query.dtype}
        return _CakeNVFP4CorrectionWorkspace(
            q_mean=torch.empty((batch * heads, q_blocks, head_dim), **common),
            k_mean=torch.empty((batch * heads, head_dim), **common),
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
            torch.cuda.current_stream(query.device).cuda_stream,
        )
        if key != self._workspace_key:
            with _SHARED_SCRATCH_LOCK:
                shared_scratch = _SHARED_SCRATCH.get(key)
                if shared_scratch is None:
                    allocated_workspace = (
                        allocate_cake_nvfp4_attention_workspace(
                            query, qkv_layout="NHD"
                        )
                    )
                    shared_scratch = _CakeNVFP4SharedScratch(
                        packed=allocated_workspace.packed,
                        correction=self._allocate_correction_workspace(query),
                    )
                    _SHARED_SCRATCH[key] = shared_scratch
            self._shared_scratch = shared_scratch
            self._output = torch.empty_like(query, dtype=torch.bfloat16)
            self._workspace_key = key
        return (
            self._shared_scratch.packed,
            self._output,
            self._shared_scratch.correction,
        )

    @staticmethod
    def _prepare_qk_correction(
        query: torch.Tensor,
        key: torch.Tensor,
        packed,
        workspace: _CakeNVFP4CorrectionWorkspace,
    ) -> torch.Tensor:
        batch, seq_len, heads, _ = query.shape
        padded_seq_len = packed.q_fp4.shape[2]
        q_blocks = padded_seq_len // 128

        from loom.examples.weave.fp4_attention_quantize import (
            make_centered_qk_launch,
        )

        make_centered_qk_launch(
            query,
            key,
            outputs=(
                packed.q_fp4,
                packed.k_fp4,
                packed.q_scale,
                packed.k_scale,
                workspace.q_mean,
                workspace.k_mean,
            ),
            qkv_layout="NHD",
        )()
        workspace.qk_correction.zero_()
        torch.bmm(
            workspace.q_mean,
            key.permute(0, 2, 3, 1).squeeze(0),
            out_dtype=torch.float32,
            out=workspace.qk_correction.squeeze(0)[..., :seq_len],
        )

        return workspace.qk_correction.view(
            batch * heads, q_blocks, padded_seq_len
        )

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
            from flashinfer import cake_nvfp4_attention_fwd
            from flashinfer.cake_nvfp4_attention import (
                _get_module,
                _target_for_device,
            )
        except ImportError as error:
            raise ImportError(
                "Cake NVFP4 attention requires a FlashInfer build that exports "
                "the Cake packed attention API."
            ) from error

        packed, output, correction_workspace = self._get_workspace_and_output(query)
        qk_correction = self._prepare_qk_correction(
            query, key, packed, correction_workspace
        )
        quantize_module = _get_module(_target_for_device(value.device), False)
        quantize_module.quantize_v(
            value,
            packed.v_fp4_t,
            packed.v_scale_lo,
            packed.v_scale_hi,
            0,
        )
        return cake_nvfp4_attention_fwd(
            *packed,
            seq_len,
            sm_scale=self.softmax_scale,
            causal=False,
            qkv_layout="NHD",
            qk_correction=qk_correction,
            out=output,
        )
