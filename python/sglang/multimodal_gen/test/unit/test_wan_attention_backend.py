import unittest
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanSelfAttention,
    WanTransformer3DModel,
    _wan_cross_attention_backends,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def test_cake_nvfp4_is_admitted_for_wan_self_attention_only(self):
        self.assertIn(
            AttentionBackendEnum.CAKE_NVFP4,
            WanTransformer3DModel._supported_attention_backends,
        )
        cross = _wan_cross_attention_backends(
            {
                AttentionBackendEnum.CAKE_NVFP4,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        )
        self.assertEqual(
            cross,
            {AttentionBackendEnum.FA},
        )

    def test_non_cake_cross_attention_preserves_dense_candidates(self):
        cross = _wan_cross_attention_backends(
            {
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        )
        self.assertEqual(
            cross,
            {AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

    def test_cross_attention_role_is_forwarded_to_usp(self):
        with (
            patch(f"{_WAN}.ColumnParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.RowParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.get_tp_world_size", return_value=1),
            patch(f"{_WAN}.USPAttention") as usp_attention,
        ):
            WanSelfAttention(
                dim=128,
                num_heads=1,
                qk_norm=False,
                is_cross_attention=True,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
            )

        self.assertTrue(usp_attention.call_args.kwargs["is_cross_attention"])
        self.assertTrue(usp_attention.call_args.kwargs["skip_sequence_parallel"])


if __name__ == "__main__":
    unittest.main()
