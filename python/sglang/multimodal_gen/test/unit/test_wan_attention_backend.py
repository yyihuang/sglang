import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanSelfAttention,
    WanTransformer3DModel,
    _is_wan_hybrid_teacher_forced_timestep,
    _run_wan_hybrid_abba_benchmark,
    _run_wan_hybrid_teacher_forced_attention,
    _use_wan_hybrid_for_request_timestep,
    _use_wan_hybrid_for_timestep,
    _validate_wan_hybrid_abba_activity,
    _validate_wan_hybrid_layer_indices,
    _validate_wan_hybrid_min_timestep,
    _validate_wan_hybrid_teacher_forced_compare,
    _validate_wan_hybrid_teacher_forced_timestep,
    _wan_hybrid_abba_side_recorded,
    _wan_hybrid_abba_timing_payload,
    _wan_hybrid_teacher_forced_tensor_metrics,
    _wan_cross_attention_backends,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    _validate_wan_hybrid_abba_configuration,
    _validate_wan_hybrid_abba_records,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def test_abba_timing_payload_requires_five_samples(self):
        activities = [
            {"kernel_activities": [{"name": "kernel"}]} for _ in range(5)
        ]
        payload = _wan_hybrid_abba_timing_payload(
            [0.5, 0.4, 0.6, 0.3, 0.7], activities
        )

        self.assertEqual(payload["backend"], "cupti")
        self.assertEqual(payload["median_ms"], 0.5)
        self.assertEqual(
            payload["cupti_activity_evidence"]["sample_count"], 5
        )
        with self.assertRaisesRegex(RuntimeError, "expected 5 samples"):
            _wan_hybrid_abba_timing_payload([0.5], activities[:1])

    def test_abba_activity_requires_exact_candidate_and_fa4_surfaces(self):
        candidate = [
            {
                "kernel_activities": [
                    {"name": "kernel_wan_hybrid_quantize_value"},
                    {"name": "kernel_wan_hybrid_attention"},
                ]
            }
        ]
        fa4 = [{"kernel_activities": [{"name": "flash_fwd"}]}]
        _validate_wan_hybrid_abba_activity("candidate", candidate)
        _validate_wan_hybrid_abba_activity("fa4", fa4)

        with self.assertRaisesRegex(RuntimeError, "quantization followed"):
            _validate_wan_hybrid_abba_activity("candidate", fa4)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _validate_wan_hybrid_abba_activity("fa4", candidate)

    def test_abba_side_scoping_records_each_cfg_side_once(self):
        metrics = RequestMetrics("request")
        self.assertFalse(_wan_hybrid_abba_side_recorded(metrics, False))
        metrics.wan_hybrid_abba_benchmarks.append({"cfg_negative": False})
        self.assertTrue(_wan_hybrid_abba_side_recorded(metrics, False))
        self.assertFalse(_wan_hybrid_abba_side_recorded(metrics, True))
        metrics.wan_hybrid_abba_benchmarks.append({"cfg_negative": True})
        self.assertTrue(_wan_hybrid_abba_side_recorded(metrics, True))

    def test_abba_runs_both_orders_on_the_same_qkv_and_stream(self):
        calls = []

        class Attention(nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind

            def forward(self, query, key, value):
                calls.append((self.kind, query, key, value))
                return value

        reference = Attention("fa4")
        candidate = Attention("candidate")
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 2, 1, 4)
        value = torch.randn(1, 2, 1, 4)
        stream = SimpleNamespace(cuda_stream=123)
        properties = SimpleNamespace(
            name="B200",
            uuid="device-uuid",
            major=10,
            minor=0,
            total_memory=1,
        )

        def fake_bench(fn):
            fn()
            elapsed = 1.0 if calls[-1][0] == "fa4" else 2.0
            kernel_names = (
                ["flash_fwd"]
                if calls[-1][0] == "fa4"
                else [
                    "kernel_wan_hybrid_quantize_value",
                    "kernel_wan_hybrid_attention",
                ]
            )
            activity_evidence = [
                {
                    "kernel_activities": [
                        {"name": name} for name in kernel_names
                    ]
                }
                for _ in range(5)
            ]
            return [elapsed] * 5, activity_evidence

        with (
            patch(f"{_WAN}._bench_wan_hybrid_attention", side_effect=fake_bench),
            patch(f"{_WAN}.torch.cuda.current_stream", return_value=stream),
            patch(f"{_WAN}.torch.cuda.current_device", return_value=0),
            patch(f"{_WAN}.torch.cuda.get_device_properties", return_value=properties),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "flash_attn.fa_ver",
                4,
            ),
        ):
            output, record = _run_wan_hybrid_abba_benchmark(
                reference, candidate, query, key, value
            )

        self.assertIs(output, value)
        self.assertEqual(
            [kind for kind, *_ in calls],
            [
                "candidate",
                "fa4",
                "fa4",
                "candidate",
                "fa4",
                "candidate",
                "candidate",
                "fa4",
                "candidate",
            ],
        )
        for _, observed_q, observed_k, observed_v in calls:
            self.assertIs(observed_q, query)
            self.assertIs(observed_k, key)
            self.assertIs(observed_v, value)
        self.assertEqual(record["stream"], 123)
        self.assertEqual(record["query_id"], id(query))
        self.assertEqual(record["key_id"], id(key))
        self.assertEqual(record["value_id"], id(value))
        self.assertEqual(
            record["timing_primitive"], "flashinfer.testing.bench_gpu_time"
        )

    def test_abba_finalization_requires_one_record_per_cfg_side(self):
        metrics = RequestMetrics("request")
        batch = SimpleNamespace(
            wan_hybrid_abba_benchmark=True,
            do_classifier_free_guidance=True,
            metrics=metrics,
        )

        with self.assertRaisesRegex(RuntimeError, "exactly one record"):
            _validate_wan_hybrid_abba_records(batch)

        metrics.wan_hybrid_abba_benchmarks.extend(
            [{"cfg_negative": False}, {"cfg_negative": True}]
        )
        _validate_wan_hybrid_abba_records(batch)

        metrics.wan_hybrid_abba_benchmarks.append({"cfg_negative": True})
        with self.assertRaisesRegex(RuntimeError, "exactly one record"):
            _validate_wan_hybrid_abba_records(batch)

    def test_abba_rejects_cfg_parallel_explicitly(self):
        batch = SimpleNamespace(
            wan_hybrid_abba_benchmark=True,
            do_classifier_free_guidance=True,
        )

        with self.assertRaisesRegex(RuntimeError, "requires serial CFG"):
            _validate_wan_hybrid_abba_configuration(
                batch, SimpleNamespace(enable_cfg_parallel=True)
            )

        batch.do_classifier_free_guidance = False
        _validate_wan_hybrid_abba_configuration(
            batch, SimpleNamespace(enable_cfg_parallel=True)
        )

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

    def test_teacher_forced_request_routes_only_the_exact_timestep(self):
        for timestep in (999, 937, 214):
            with self.subTest(timestep=timestep):
                self.assertEqual(
                    _use_wan_hybrid_for_request_timestep(
                        torch.tensor([timestep]),
                        214.0,
                        teacher_forced_request=True,
                        teacher_forced_timestep=937.0,
                    ),
                    timestep == 937,
                )

    def test_non_teacher_forced_request_preserves_threshold_routing(self):
        self.assertTrue(
            _use_wan_hybrid_for_request_timestep(
                torch.tensor([999]),
                937.0,
                teacher_forced_request=False,
                teacher_forced_timestep=937.0,
            )
        )
        self.assertFalse(
            _use_wan_hybrid_for_request_timestep(
                torch.tensor([899]),
                937.0,
                teacher_forced_request=False,
                teacher_forced_timestep=937.0,
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
        fa_output, wan_output, wan_repeat_output = (
            _run_wan_hybrid_teacher_forced_attention(
                Attention(0.0), Attention(1.0), query, key, value
            )
        )

        self.assertIs(calls[0][0], query)
        self.assertIs(calls[0][1], key)
        self.assertIs(calls[0][2], value)
        self.assertIs(calls[1][0], query)
        self.assertIs(calls[1][1], key)
        self.assertIs(calls[1][2], value)
        self.assertIs(calls[2][0], query)
        self.assertIs(calls[2][1], key)
        self.assertIs(calls[2][2], value)
        torch.testing.assert_close(fa_output, value)
        torch.testing.assert_close(wan_output, value + 1.0)
        torch.testing.assert_close(wan_repeat_output, value + 1.0)

    def test_teacher_forced_repeat_preserves_reused_candidate_output(self):
        class ReferenceAttention(nn.Module):
            def forward(self, query, key, value):
                return value

        class ReusingAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.zeros(1)
                self.calls = 0

            def forward(self, query, key, value):
                self.calls += 1
                return self.output.fill_(self.calls)

        candidate = ReusingAttention()
        _, first, second = _run_wan_hybrid_teacher_forced_attention(
            ReferenceAttention(),
            candidate,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )

        torch.testing.assert_close(first, torch.ones(1))
        torch.testing.assert_close(second, torch.full((1,), 2.0))

    def test_teacher_forced_tensor_metrics_report_local_error(self):
        reference = torch.tensor([1.0, 2.0, 3.0])
        candidate = torch.tensor([1.0, 2.5, 2.0])

        metrics = _wan_hybrid_teacher_forced_tensor_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["max_abs"], 1.0)
        self.assertTrue(metrics["finite"])
        self.assertTrue(metrics["within_tolerance"])
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
