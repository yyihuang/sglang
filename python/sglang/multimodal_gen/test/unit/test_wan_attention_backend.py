import unittest
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanSelfAttention,
    WanTransformer3DModel,
    _use_cake_nvfp4_for_timestep,
    _validate_cake_nvfp4_min_timestep,
    _wan_cross_attention_backends,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def test_cake_nvfp4_timestep_threshold(self):
        self.assertTrue(
            _use_cake_nvfp4_for_timestep(torch.tensor([999]), 975.0)
        )
        self.assertTrue(
            _use_cake_nvfp4_for_timestep(torch.tensor([975]), 975.0)
        )
        self.assertFalse(
            _use_cake_nvfp4_for_timestep(torch.tensor([972]), 975.0)
        )
        self.assertTrue(
            _use_cake_nvfp4_for_timestep(torch.tensor([0]), None)
        )

    def test_cake_nvfp4_timestep_threshold_validation(self):
        self.assertEqual(_validate_cake_nvfp4_min_timestep(975), 975.0)
        self.assertIsNone(_validate_cake_nvfp4_min_timestep(None))
        for invalid in (True, "975", -1, 1001, float("inf")):
            with self.subTest(invalid=invalid), pytest.raises(ValueError):
                _validate_cake_nvfp4_min_timestep(invalid)

    def test_cake_nvfp4_is_admitted_for_wan_self_attention_only(self):
        self.assertIn(
            AttentionBackendEnum.CAKE_NVFP4,
            WanTransformer3DModel._supported_attention_backends,
        )
        with (
            patch(f"{_WAN}.get_global_forced_attn_backend", return_value=None),
            patch(
                f"{_WAN}.get_component_forced_attn_backend",
                return_value=AttentionBackendEnum.CAKE_NVFP4,
            ),
        ):
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
        with patch(
            f"{_WAN}.get_global_forced_attn_backend",
            return_value=AttentionBackendEnum.FA,
        ):
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
