import unittest
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanSelfAttention,
    WanTransformer3DModel,
    _is_wan_hybrid_teacher_forced_timestep,
    _run_wan_hybrid_teacher_forced_attention,
    _use_wan_hybrid_for_timestep,
    _validate_wan_hybrid_layer_indices,
    _validate_wan_hybrid_min_timestep,
    _validate_wan_hybrid_teacher_forced_compare,
    _validate_wan_hybrid_teacher_forced_timestep,
    _wan_hybrid_teacher_forced_tensor_metrics,
    _wan_cross_attention_backends,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def test_wan_hybrid_timestep_threshold(self):
        self.assertTrue(_use_wan_hybrid_for_timestep(torch.tensor([999]), 975.0))
        self.assertTrue(_use_wan_hybrid_for_timestep(torch.tensor([975]), 975.0))
        self.assertFalse(_use_wan_hybrid_for_timestep(torch.tensor([972]), 975.0))
        self.assertTrue(_use_wan_hybrid_for_timestep(torch.tensor([0]), None))

    def test_wan_hybrid_timestep_threshold_validation(self):
        self.assertEqual(_validate_wan_hybrid_min_timestep(975), 975.0)
        self.assertIsNone(_validate_wan_hybrid_min_timestep(None))
        for invalid in (True, "975", -1, 1001, float("inf")):
            with self.subTest(invalid=invalid), pytest.raises(ValueError):
                _validate_wan_hybrid_min_timestep(invalid)

    def test_wan_hybrid_layer_indices_validation(self):
        self.assertEqual(
            _validate_wan_hybrid_layer_indices([0, 39], 40),
            frozenset({0, 39}),
        )
        self.assertEqual(_validate_wan_hybrid_layer_indices([], 40), frozenset())
        self.assertIsNone(_validate_wan_hybrid_layer_indices(None, 40))
        for invalid in (True, 1, "0", [False], [1.0], [-1], [40], [3, 3]):
            with self.subTest(invalid=invalid), pytest.raises(ValueError):
                _validate_wan_hybrid_layer_indices(invalid, 40)

    def test_wan_hybrid_teacher_forced_config_validation(self):
        self.assertFalse(_validate_wan_hybrid_teacher_forced_compare(None))
        self.assertTrue(_validate_wan_hybrid_teacher_forced_compare(True))
        for invalid in (0, 1, "true", []):
            with self.subTest(invalid=invalid), pytest.raises(ValueError):
                _validate_wan_hybrid_teacher_forced_compare(invalid)

        self.assertEqual(_validate_wan_hybrid_teacher_forced_timestep(None), 999.0)
        self.assertEqual(_validate_wan_hybrid_teacher_forced_timestep(750), 750.0)
        for invalid in (True, "999", -1, 1001, float("inf")):
            with self.subTest(invalid=invalid), pytest.raises(ValueError):
                _validate_wan_hybrid_teacher_forced_timestep(invalid)

    def test_wan_hybrid_teacher_forced_timestep_requires_exact_match(self):
        self.assertTrue(
            _is_wan_hybrid_teacher_forced_timestep(torch.tensor([999]), 999.0)
        )
        self.assertFalse(
            _is_wan_hybrid_teacher_forced_timestep(torch.tensor([998]), 999.0)
        )
        self.assertFalse(
            _is_wan_hybrid_teacher_forced_timestep(
                torch.tensor([999, 998]), 999.0
            )
        )

    def test_teacher_forced_attention_shares_qkv_and_keeps_fa_live(self):
        calls = []

        class Attention(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, query, key, value):
                calls.append((query, key, value))
                return value + self.offset

        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 2, 1, 4)
        value = torch.randn(1, 2, 1, 4)
        fa_output, wan_output = _run_wan_hybrid_teacher_forced_attention(
            Attention(0.0), Attention(1.0), query, key, value
        )

        self.assertIs(calls[0][0], query)
        self.assertIs(calls[0][1], key)
        self.assertIs(calls[0][2], value)
        self.assertIs(calls[1][0], query)
        self.assertIs(calls[1][1], key)
        self.assertIs(calls[1][2], value)
        torch.testing.assert_close(fa_output, value)
        torch.testing.assert_close(wan_output, value + 1.0)

    def test_teacher_forced_tensor_metrics_report_local_error(self):
        reference = torch.tensor([1.0, 2.0, 3.0])
        candidate = torch.tensor([1.0, 2.5, 2.0])

        metrics = _wan_hybrid_teacher_forced_tensor_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["max_abs"], 1.0)
        self.assertTrue(metrics["finite"])
        self.assertFalse(metrics["exact_match"])

    def test_wan_hybrid_is_admitted_for_wan_self_attention_only(self):
        self.assertIn(
            AttentionBackendEnum.WAN_HYBRID,
            WanTransformer3DModel._supported_attention_backends,
        )
        with (
            patch(f"{_WAN}.get_global_forced_attn_backend", return_value=None),
            patch(
                f"{_WAN}.get_component_forced_attn_backend",
                return_value=AttentionBackendEnum.WAN_HYBRID,
            ),
        ):
            cross = _wan_cross_attention_backends(
                {
                    AttentionBackendEnum.WAN_HYBRID,
                    AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                }
            )
        self.assertEqual(
            cross,
            {AttentionBackendEnum.FA},
        )

    def test_non_wan_hybrid_cross_attention_preserves_dense_candidates(self):
        with patch(
            f"{_WAN}.get_global_forced_attn_backend",
            return_value=AttentionBackendEnum.FA,
        ):
            cross = _wan_cross_attention_backends(
                {
                    AttentionBackendEnum.WAN_HYBRID,
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
