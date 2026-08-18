# SPDX-License-Identifier: Apache-2.0

import functools
import math
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
    qk_correction: torch.Tensor


class _CakeNVFP4WanWorkspace(NamedTuple):
    q_rstd: torch.Tensor
    k_rstd: torch.Tensor
    q_mean_fp4: torch.Tensor
    q_mean_scale: torch.Tensor
    dummy_correction: torch.Tensor


@dataclass
class _CakeNVFP4SharedScratch:
    packed: object
    correction: _CakeNVFP4CorrectionWorkspace
    wan: _CakeNVFP4WanWorkspace | None = None


_SHARED_SCRATCH_LOCK = threading.Lock()
_SHARED_SCRATCH: weakref.WeakValueDictionary[tuple, _CakeNVFP4SharedScratch] = (
    weakref.WeakValueDictionary()
)


@functools.cache
def _get_wan_fp4_attention_module(device_index: int):
    from loom.examples.weave.fp4_flash_attention import (
        fp4_flash_attention_qk_fp4_corrected,
    )
    from loom.runtime import build_kernel_module

    with torch.cuda.device(device_index):
        return build_kernel_module(
            fp4_flash_attention_qk_fp4_corrected,
            validate=False,
            options=["--use_fast_math"],
            IS_CAUSAL=0,
            HAS_QK_CORRECTION=0,
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
            qk_correction=torch.empty(
                (batch, heads, q_blocks, padded_seq_len),
                device=query.device,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _allocate_wan_workspace(
        query: torch.Tensor,
        packed,
    ) -> _CakeNVFP4WanWorkspace:
        batch, seq_len, _, _ = query.shape
        return _CakeNVFP4WanWorkspace(
            q_rstd=torch.empty(
                (batch * seq_len,), device=query.device, dtype=torch.float32
            ),
            k_rstd=torch.empty(
                (batch * seq_len,), device=query.device, dtype=torch.float32
            ),
            q_mean_fp4=torch.empty_like(packed.q_fp4),
            q_mean_scale=torch.empty_like(packed.q_scale),
            dummy_correction=torch.zeros(
                (1,), device=query.device, dtype=torch.float32
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

    def _get_wan_workspace(self, query: torch.Tensor, packed):
        with _SHARED_SCRATCH_LOCK:
            if self._shared_scratch.wan is None:
                self._shared_scratch.wan = self._allocate_wan_workspace(
                    query, packed
                )
        return self._shared_scratch.wan

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
            make_q_centered_qk_launch,
        )

        make_q_centered_qk_launch(
            query,
            key,
            outputs=(
                packed.q_fp4,
                packed.k_fp4,
                packed.q_scale,
                packed.k_scale,
                workspace.q_mean,
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
        seq_len = query.shape[1]
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

    def forward_wan_projections(
        self,
        query_projection: torch.Tensor,
        key_projection: torch.Tensor,
        value_projection: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        """Run the qualified Wan projection-to-attention fused path."""

        expected_projection_shape = (1, query_projection.shape[1], 5120)
        if (
            query_projection.ndim != 3
            or tuple(query_projection.shape) != expected_projection_shape
            or key_projection.shape != query_projection.shape
            or value_projection.shape != query_projection.shape
        ):
            raise ValueError(
                "Cake fused Wan serving requires Q/K/V projections with shape "
                f"[1, seq_len, 5120], got {tuple(query_projection.shape)}, "
                f"{tuple(key_projection.shape)}, and {tuple(value_projection.shape)}"
            )
        if self.num_heads != 40 or self.head_size != 128:
            raise ValueError(
                "Cake fused Wan serving requires H40/D128, got "
                f"H{self.num_heads}/D{self.head_size}"
            )
        for name, tensor in (
            ("query_projection", query_projection),
            ("key_projection", key_projection),
            ("value_projection", value_projection),
        ):
            if (
                tensor.device.type != "cuda"
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous BF16 CUDA storage"
                )
        seq_len = int(query_projection.shape[1])
        for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
            if (
                tuple(weight.shape) != (5120,)
                or weight.device != query_projection.device
                or weight.dtype != torch.bfloat16
                or not weight.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous BF16 [5120] on the projection device"
                )
        if (
            tuple(cos_sin_cache.shape) != (seq_len, 128)
            or cos_sin_cache.device != query_projection.device
            or cos_sin_cache.dtype != torch.float32
            or not cos_sin_cache.is_contiguous()
        ):
            raise ValueError(
                "cos_sin_cache must be contiguous FP32 [seq_len, 128] on the "
                "projection device"
            )

        query = query_projection.view(1, seq_len, 40, 128)
        packed, output, _ = self._get_workspace_and_output(query)
        wan = self._get_wan_workspace(query, packed)

        from loom.examples.weave.fp4_attention_wan_fused import (
            make_wan_qk_norm_rope_pack_launch,
            make_wan_qk_rstd_launch,
        )

        make_wan_qk_rstd_launch(
            query_projection,
            key_projection,
            outputs=(wan.q_rstd, wan.k_rstd),
            eps=eps,
        )()
        make_wan_qk_norm_rope_pack_launch(
            query_projection,
            key_projection,
            value_projection,
            q_weight,
            k_weight,
            wan.q_rstd,
            wan.k_rstd,
            cos_sin_cache,
            outputs=(
                packed.q_fp4,
                packed.k_fp4,
                packed.q_scale,
                packed.k_scale,
                wan.q_mean_fp4,
                wan.q_mean_scale,
                packed.v_fp4_t,
                packed.v_scale_lo,
                packed.v_scale_hi,
            ),
        )()

        padded_seq_len = int(packed.q_fp4.shape[2])
        total_bh = self.num_heads
        total_tiles = total_bh * ((seq_len + 511) // 512)
        num_sms = torch.cuda.get_device_properties(
            query_projection.device
        ).multi_processor_count
        persistent_clusters = min(num_sms // 2, total_tiles)
        module = _get_wan_fp4_attention_module(query_projection.device.index or 0)
        module.launch(
            grid=(2 * persistent_clusters, 1, 1),
            Q=packed.q_fp4.reshape(total_bh * padded_seq_len, 64),
            QMean=wan.q_mean_fp4.reshape(total_bh * padded_seq_len, 64),
            K=packed.k_fp4.reshape(total_bh * padded_seq_len, 64),
            Vt=packed.v_fp4_t.reshape(total_bh * 128, padded_seq_len // 2),
            SFQ=packed.q_scale,
            SFQMean=wan.q_mean_scale,
            SFK=packed.k_scale,
            SFVtLo=packed.v_scale_lo,
            SFVtHi=packed.v_scale_hi,
            QKCorrection=wan.dummy_correction,
            O=output.permute(0, 2, 1, 3),
            seqlen_q=seq_len,
            seqlen_kv=seq_len,
            q_stride=padded_seq_len,
            kv_stride=padded_seq_len,
            softmax_scale_log2=self.softmax_scale / math.log(2.0),
            heads=self.num_heads,
            total_bh=total_bh,
        )
        return output
