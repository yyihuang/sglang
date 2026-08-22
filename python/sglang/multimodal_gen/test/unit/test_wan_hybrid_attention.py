# SPDX-License-Identifier: Apache-2.0

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    WanHybridAttentionBackend,
    WanHybridEvidenceCollector,
    WanHybridAttentionImpl,
    _SHARED_SCRATCH,
    _record_successful_wan_hybrid_forward,
    read_wan_hybrid_hit_count,
    record_wan_attention_route,
    reset_wan_hybrid_hit_count,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _exact_cuda_inputs():
    device = SimpleNamespace(type="cuda", index=0)
    tensors = []
    for _ in range(3):
        tensor = Mock()
        tensor.shape = (1, 4800, 40, 128)
        tensor.ndim = 4
        tensor.device = device
        tensor.dtype = torch.bfloat16
        tensor.is_contiguous.return_value = True
        tensors.append(tensor)
    return tuple(tensors)


def _make_impl(*, prefix=""):
    return WanHybridAttentionImpl(
        num_heads=40,
        num_kv_heads=40,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
        prefix=prefix,
    )


class TestWanHybridAttentionBackend(unittest.TestCase):
    def setUp(self):
        reset_wan_hybrid_hit_count()

    def test_backend_contract(self):
        self.assertEqual(
            WanHybridAttentionBackend.get_enum(), AttentionBackendEnum.WAN_HYBRID
        )
        self.assertEqual(WanHybridAttentionBackend.get_supported_head_sizes(), [128])
        self.assertFalse(WanHybridAttentionBackend.supports_packed_varlen())
        self.assertFalse(WanHybridAttentionBackend.supports_ring_rotation())

    def test_constructor_accepts_exact_wan_self_attention(self):
        self.assertEqual(_make_impl().num_heads, 40)
        self.assertEqual(
            _make_impl(prefix="blocks.17.attn1.impl")._wan_layer_index, 17
        )

    def test_request_coverage_uses_real_context_and_route_events(self):
        result = torch.ones(1)
        collector = WanHybridEvidenceCollector(request_id="request-a")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            wan_component_name="transformer",
            wan_actual_timestep=999,
            wan_cfg_branch_index=0,
            wan_hybrid_evidence_collector=collector,
        ):
            for layer_index in range(2):
                record_wan_attention_route(
                    layer_index=layer_index,
                    hybrid_configured=True,
                    eligible_for_hybrid=True,
                )
                _record_successful_wan_hybrid_forward(
                    result, layer_index=layer_index
                )
        with set_forward_context(
            current_timestep=1,
            attn_metadata=None,
            wan_component_name="transformer_2",
            wan_actual_timestep=100,
            wan_cfg_branch_index=0,
            wan_hybrid_evidence_collector=collector,
        ):
            for layer_index in range(2):
                record_wan_attention_route(
                    layer_index=layer_index,
                    hybrid_configured=False,
                    eligible_for_hybrid=False,
                )

        coverage = collector.coverage()

        self.assertEqual(coverage["schema_version"], 2)
        self.assertEqual(coverage["request_id"], "request-a")
        self.assertEqual(coverage["expected_hit_count"], 2)
        self.assertEqual(coverage["actual_hit_count"], 2)
        self.assertEqual(coverage["unattributed_actual_hit_count"], 0)
        self.assertEqual(
            coverage["steps"][0]["branches"][0][
                "planned_hybrid_layer_indices"
            ],
            [0, 1],
        )
        self.assertEqual(
            coverage["steps"][1]["branches"][0]["control_layer_indices"],
            [0, 1],
        )

    def test_interleaved_collectors_are_isolated_and_misses_are_real(self):
        result = torch.ones(1)
        first = WanHybridEvidenceCollector(request_id="request-a")
        second = WanHybridEvidenceCollector(request_id="request-b")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            wan_component_name="transformer",
            wan_actual_timestep=999,
            wan_cfg_branch_index=0,
            wan_hybrid_evidence_collector=first,
        ):
            record_wan_attention_route(
                layer_index=0,
                hybrid_configured=True,
                eligible_for_hybrid=True,
            )
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                wan_component_name="transformer",
                wan_actual_timestep=999,
                wan_cfg_branch_index=0,
                wan_hybrid_evidence_collector=second,
            ):
                record_wan_attention_route(
                    layer_index=0,
                    hybrid_configured=True,
                    eligible_for_hybrid=True,
                )
                record_wan_attention_route(
                    layer_index=1,
                    hybrid_configured=True,
                    eligible_for_hybrid=False,
                )
            _record_successful_wan_hybrid_forward(result, layer_index=0)

        first_coverage = first.coverage()
        second_coverage = second.coverage()
        self.assertEqual(first_coverage["actual_hit_count"], 1)
        self.assertEqual(first_coverage["eligible_hybrid_miss_count"], 0)
        self.assertEqual(second_coverage["actual_hit_count"], 0)
        self.assertEqual(second_coverage["eligible_hybrid_miss_count"], 1)
        second_branch = second_coverage["steps"][0]["branches"][0]
        self.assertEqual(second_branch["eligible_hybrid_miss_layer_indices"], [0])
        self.assertEqual(second_branch["configured_fallback_layer_indices"], [1])

    def test_constructor_rejects_unsupported_contracts(self):
        cases = (
            ({"head_size": 64}, "head_size=128"),
            ({"causal": True}, "noncausal"),
            ({"num_kv_heads": 1}, "equal query and KV"),
            ({"num_heads": 32, "num_kv_heads": 32}, "num_heads=40"),
        )
        defaults = {
            "num_heads": 40,
            "num_kv_heads": 40,
            "head_size": 128,
            "softmax_scale": 128**-0.5,
            "causal": False,
        }
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                WanHybridAttentionImpl(**(defaults | overrides))

    def test_forward_uses_only_public_api_and_reuses_buffers(self):
        impl = _make_impl()
        query, key, value = _exact_cuda_inputs()
        workspace = object()
        output = object()
        workspace_type = Mock(return_value=workspace)
        is_available = Mock(return_value=True)
        run = Mock(side_effect=lambda *_args, **kwargs: kwargs["out"])
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.WanHybridAttentionWorkspace = workspace_type
        fake_flashinfer.is_wan_hybrid_attention_available = is_available
        fake_flashinfer.wan_hybrid_attention = run

        with (
            patch.dict(_SHARED_SCRATCH, {}, clear=True),
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "wan_hybrid.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=17),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "wan_hybrid.torch.empty_like",
                return_value=output,
            ) as empty_like,
        ):
            first = impl.forward(query, key, value, None)
            second = impl.forward(query, key, value, None)

        self.assertIs(first, output)
        self.assertIs(second, output)
        workspace_type.assert_called_once_with(query.device)
        empty_like.assert_called_once_with(query, dtype=torch.bfloat16)
        self.assertEqual(is_available.call_count, 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(read_wan_hybrid_hit_count(), 2)
        for call in run.call_args_list:
            self.assertEqual(call.args, (query, key, value))
            self.assertIs(call.kwargs["out"], output)
            self.assertIs(call.kwargs["workspace"], workspace)
            self.assertEqual(call.kwargs["sm_scale"], 128**-0.5)
            self.assertEqual(call.kwargs["qkv_layout"], "NHD")
            self.assertFalse(call.kwargs["causal"])

    def test_successful_public_forward_is_attributed_to_active_request(self):
        impl = _make_impl(prefix="blocks.7.attn1.impl")
        query, key, value = _exact_cuda_inputs()
        output = object()
        collector = WanHybridEvidenceCollector(request_id="request-public-api")
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.is_wan_hybrid_attention_available = Mock(return_value=True)
        fake_flashinfer.wan_hybrid_attention = Mock(return_value=output)

        with (
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch.object(
                impl,
                "_get_workspace_and_output",
                return_value=(object(), output),
            ),
            set_forward_context(
                current_timestep=3,
                attn_metadata=None,
                wan_component_name="transformer",
                wan_actual_timestep=777,
                wan_cfg_branch_index=1,
                wan_hybrid_evidence_collector=collector,
            ),
        ):
            record_wan_attention_route(
                layer_index=7,
                hybrid_configured=True,
                eligible_for_hybrid=True,
            )
            result = impl.forward(query, key, value, None)

        coverage = collector.coverage()
        self.assertIs(result, output)
        self.assertEqual(coverage["expected_hit_count"], 1)
        self.assertEqual(coverage["actual_hit_count"], 1)
        self.assertEqual(coverage["unattributed_actual_hit_count"], 0)
        self.assertEqual(read_wan_hybrid_hit_count(), 0)
        fake_flashinfer.wan_hybrid_attention.assert_called_once()

    def test_unavailable_public_backend_fails_closed_without_hit(self):
        impl = _make_impl()
        query, key, value = _exact_cuda_inputs()
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.WanHybridAttentionWorkspace = Mock()
        fake_flashinfer.is_wan_hybrid_attention_available = Mock(return_value=False)
        fake_flashinfer.wan_hybrid_attention = Mock()

        with (
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            self.assertRaisesRegex(NotImplementedError, "unavailable"),
        ):
            impl.forward(query, key, value, None)

        fake_flashinfer.wan_hybrid_attention.assert_not_called()
        self.assertEqual(read_wan_hybrid_hit_count(), 0)

    def test_failed_public_forward_does_not_increment_hit_count(self):
        impl = _make_impl()
        query, key, value = _exact_cuda_inputs()
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.is_wan_hybrid_attention_available = Mock(return_value=True)
        fake_flashinfer.wan_hybrid_attention = Mock(
            side_effect=RuntimeError("kernel failed")
        )

        with (
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch.object(
                impl,
                "_get_workspace_and_output",
                return_value=(object(), object()),
            ),
            self.assertRaisesRegex(RuntimeError, "kernel failed"),
        ):
            impl.forward(query, key, value, None)

        self.assertEqual(read_wan_hybrid_hit_count(), 0)

    def test_workspace_is_shared_across_layers_on_same_stream(self):
        first = _make_impl()
        second = _make_impl()
        query, _, _ = _exact_cuda_inputs()
        workspace = object()
        workspace_type = Mock(return_value=workspace)
        fake_flashinfer = SimpleNamespace(WanHybridAttentionWorkspace=workspace_type)

        with (
            patch.dict(_SHARED_SCRATCH, {}, clear=True),
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "wan_hybrid.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=23),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "wan_hybrid.torch.empty_like",
                side_effect=(object(), object()),
            ),
        ):
            first_workspace, first_output = first._get_workspace_and_output(query)
            second_workspace, second_output = second._get_workspace_and_output(query)

        workspace_type.assert_called_once_with(query.device)
        self.assertIs(first_workspace, second_workspace)
        self.assertIsNot(first_output, second_output)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_injected_public_api_reuses_storage_and_replays_cuda_graph(self):
        """Exercise SGLang graph integration without the unfinished device impl."""

        device = torch.device("cuda", torch.cuda.current_device())

        class FakePublicWorkspace:
            instances = []

            def __init__(self, workspace_device):
                self.device = torch.device(workspace_device)
                self.__class__.instances.append(self)

        def fake_public_attention(
            q,
            k,
            v,
            *,
            out,
            workspace,
            sm_scale,
            qkv_layout,
            causal,
        ):
            self.assertIsInstance(workspace, FakePublicWorkspace)
            self.assertEqual(sm_scale, 128**-0.5)
            self.assertEqual(qkv_layout, "NHD")
            self.assertFalse(causal)
            self.assertEqual(q.device, k.device)
            self.assertEqual(q.device, v.device)
            out.copy_(q)
            return out

        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.WanHybridAttentionWorkspace = FakePublicWorkspace
        fake_flashinfer.is_wan_hybrid_attention_available = lambda _device: True
        fake_flashinfer.wan_hybrid_attention = fake_public_attention

        torch.manual_seed(4254)
        query, key, value = (
            torch.randn(
                (1, 4800, 40, 128),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(3)
        )
        impl = _make_impl()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        with (
            patch.dict(_SHARED_SCRATCH, {}, clear=True),
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
        ):
            with torch.cuda.stream(stream):
                first = impl.forward(query, key, value, None)
                second = impl.forward(query, key, value, None)
            stream.synchronize()
            self.assertIs(first, second)
            output_ptr = first.data_ptr()
            workspace = impl._shared_scratch.workspace
            self.assertEqual(len(FakePublicWorkspace.instances), 1)

            with torch.cuda.stream(stream):
                third = impl.forward(query, key, value, None)
            stream.synchronize()
            self.assertEqual(third.data_ptr(), output_ptr)
            self.assertIs(impl._shared_scratch.workspace, workspace)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = impl.forward(query, key, value, None)
            graph.replay()
            stream.synchronize()

            query.fill_(2)
            captured.zero_()
            torch.cuda.synchronize(device)
            graph.replay()
            stream.synchronize()

            self.assertEqual(captured.data_ptr(), output_ptr)
            self.assertTrue(torch.equal(captured, query))
            self.assertIs(impl._shared_scratch.workspace, workspace)
            self.assertEqual(len(FakePublicWorkspace.instances), 1)
            self.assertGreater(read_wan_hybrid_hit_count(), 0)

    def test_forward_rejects_non_exact_sequence(self):
        impl = _make_impl()
        query = torch.empty((1, 512, 40, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "sequence length 4800"):
            impl.forward(query, query, query, None)


if __name__ == "__main__":
    unittest.main()
