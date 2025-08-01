from __future__ import annotations

import dataclasses
import functools
import math
from functools import lru_cache, partial
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.utils import is_cuda, print_info_once

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

from sglang.srt.distributed import (
    parallel_state,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import add_prefix

ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}


@dataclasses.dataclass
class SingletonCache:
    data: Any = None

    def set_data(self, value: Any) -> None:
        self.data = value

    def get_data(self) -> Optional[Any]:
        return self.data

    def empty(self) -> bool:
        return self.get_data() is None


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_size)

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        # [b * s, head, head_size]
        output = torch.empty_like(q)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens.cuda(),
            seq_lens.cuda(),
            max_seqlen,
            is_causal=False,
        )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash3Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[Union[SingletonCache, torch.Tensor]],
        bsz: int,
        seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
                )
            cu_seqlens = cu_seqlens.get_data()

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        return output


QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        layer_norm_eps: float = 1e-06,
        **kwargs,
    ):
        super().__init__()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = world_size
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, world_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, world_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )
            self.k_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )

        if global_server_args_dict["mm_attention_backend"] is None:
            if qkv_backend is None:
                qkv_backend = "sdpa"
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        else:
            qkv_backend = global_server_args_dict["mm_attention_backend"]

        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=softmax_in_single_precision,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn"""
        q = q.flatten(1, 2)
        k = k.flatten(1, 2)

        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        q = q.unflatten(-1, (-1, self.head_size))
        k = k.unflatten(-1, (-1, self.head_size))
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        bsz, s, _ = x.shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)

            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if position_embeddings is not None:
            cos, sin = position_embeddings
            original_shape = q.shape
            # [total_tokens, head, head_size]
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, head, self.head_size)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            q = q.view(original_shape)
            k = k.view(original_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output
