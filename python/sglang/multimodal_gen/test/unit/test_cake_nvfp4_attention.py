# SPDX-License-Identifier: Apache-2.0

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.cake_nvfp4 import (
    CakeNVFP4AttentionBackend,
    CakeNVFP4AttentionImpl,
    _SHARED_SCRATCH,
    read_cake_nvfp4_hit_count,
    reset_cake_nvfp4_hit_count,
)
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


def _make_impl():
    return CakeNVFP4AttentionImpl(
        num_heads=40,
        num_kv_heads=40,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
    )


class TestCakeNVFP4AttentionBackend(unittest.TestCase):
    def setUp(self):
        reset_cake_nvfp4_hit_count()

    def test_backend_contract(self):
        self.assertEqual(
            CakeNVFP4AttentionBackend.get_enum(), AttentionBackendEnum.CAKE_NVFP4
        )
        self.assertEqual(CakeNVFP4AttentionBackend.get_supported_head_sizes(), [128])
        self.assertFalse(CakeNVFP4AttentionBackend.supports_packed_varlen())
        self.assertFalse(CakeNVFP4AttentionBackend.supports_ring_rotation())

    def test_constructor_accepts_exact_wan_self_attention(self):
        self.assertEqual(_make_impl().num_heads, 40)

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
                CakeNVFP4AttentionImpl(**(defaults | overrides))

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
                "cake_nvfp4.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=17),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "cake_nvfp4.torch.empty_like",
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
        self.assertEqual(read_cake_nvfp4_hit_count(), 2)
        for call in run.call_args_list:
            self.assertEqual(call.args, (query, key, value))
            self.assertIs(call.kwargs["out"], output)
            self.assertIs(call.kwargs["workspace"], workspace)
            self.assertEqual(call.kwargs["sm_scale"], 128**-0.5)
            self.assertEqual(call.kwargs["qkv_layout"], "NHD")
            self.assertFalse(call.kwargs["causal"])

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
        self.assertEqual(read_cake_nvfp4_hit_count(), 0)

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

        self.assertEqual(read_cake_nvfp4_hit_count(), 0)

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
                "cake_nvfp4.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=23),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "cake_nvfp4.torch.empty_like",
                side_effect=(object(), object()),
            ),
        ):
            first_workspace, first_output = first._get_workspace_and_output(query)
            second_workspace, second_output = second._get_workspace_and_output(query)

        workspace_type.assert_called_once_with(query.device)
        self.assertIs(first_workspace, second_workspace)
        self.assertIsNot(first_output, second_output)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_public_backend_reuses_storage_and_replays_cuda_graph(self):
        try:
            from flashinfer import is_wan_hybrid_attention_available
        except ImportError:
            self.skipTest("FlashInfer public wan_hybrid API is not installed")
        device = torch.device("cuda", torch.cuda.current_device())
        if not is_wan_hybrid_attention_available(device):
            self.skipTest("FlashInfer wan_hybrid implementation is unavailable")

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

        with torch.cuda.stream(stream):
            first = impl.forward(query, key, value, None)
            second = impl.forward(query, key, value, None)
        stream.synchronize()
        self.assertIs(first, second)
        output_ptr = first.data_ptr()
        workspace = impl._shared_scratch.workspace

        allocated_before = torch.cuda.memory_allocated(device)
        with torch.cuda.stream(stream):
            third = impl.forward(query, key, value, None)
        stream.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(device), allocated_before)
        self.assertEqual(third.data_ptr(), output_ptr)
        self.assertIs(impl._shared_scratch.workspace, workspace)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = impl.forward(query, key, value, None)
        graph.replay()
        stream.synchronize()
        expected = captured.clone()
        graph.replay()
        stream.synchronize()

        self.assertEqual(captured.data_ptr(), output_ptr)
        self.assertTrue(torch.equal(captured, expected))
        self.assertGreater(read_cake_nvfp4_hit_count(), 0)

    def test_forward_rejects_non_exact_sequence(self):
        impl = _make_impl()
        query = torch.empty((1, 512, 40, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "sequence length 4800"):
            impl.forward(query, query, query, None)


if __name__ == "__main__":
    unittest.main()
