# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import namedtuple
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.cake_nvfp4 import (
    CakeNVFP4AttentionBackend,
    CakeNVFP4AttentionImpl,
    _SHARED_SCRATCH,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class TestCakeNVFP4AttentionBackend(unittest.TestCase):
    def test_backend_contract(self):
        self.assertEqual(
            CakeNVFP4AttentionBackend.get_enum(), AttentionBackendEnum.CAKE_NVFP4
        )
        self.assertEqual(CakeNVFP4AttentionBackend.get_supported_head_sizes(), [128])
        self.assertFalse(CakeNVFP4AttentionBackend.supports_packed_varlen())
        self.assertFalse(CakeNVFP4AttentionBackend.supports_ring_rotation())

    def test_constructor_accepts_wan_self_attention(self):
        impl = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        self.assertEqual(impl.num_heads, 40)

    def test_correction_workspace_centers_q_only(self):
        query = torch.empty((1, 257, 4, 128), dtype=torch.bfloat16)
        workspace = CakeNVFP4AttentionImpl._allocate_correction_workspace(query)

        self.assertEqual(workspace._fields, ("q_mean", "qk_correction"))
        self.assertEqual(workspace.q_mean.shape, (4, 4, 128))
        self.assertEqual(workspace.qk_correction.shape, (1, 4, 4, 512))

    def test_wan_workspace_contains_only_reusable_serving_operands(self):
        query = torch.empty((1, 257, 40, 128), dtype=torch.bfloat16)
        packed = SimpleNamespace(
            q_fp4=torch.empty((1, 40, 512, 64), dtype=torch.uint8),
            q_scale=torch.empty((40 * 4 * 32, 32), dtype=torch.uint8),
        )

        workspace = CakeNVFP4AttentionImpl._allocate_wan_workspace(query, packed)

        self.assertEqual(
            workspace._fields,
            (
                "q_rstd",
                "k_rstd",
                "k_rope",
                "q_mean_fp4",
                "q_mean_scale",
            ),
        )
        self.assertEqual(workspace.q_rstd.shape, (257,))
        self.assertEqual(workspace.k_rstd.shape, (257,))
        self.assertEqual(workspace.k_rope.shape, (1, 257, 5120))
        self.assertEqual(workspace.q_mean_fp4.shape, packed.q_fp4.shape)
        self.assertEqual(workspace.q_mean_scale.shape, packed.q_scale.shape)

    def test_constructor_rejects_unsupported_contracts(self):
        cases = (
            ({"head_size": 64}, "head_size=128"),
            ({"causal": True}, "noncausal"),
            ({"num_kv_heads": 1}, "equal query and KV"),
        )
        defaults = {
            "num_heads": 40,
            "num_kv_heads": 40,
            "head_size": 128,
            "softmax_scale": 128**-0.5,
            "causal": False,
        }
        for overrides, message in cases:
            kwargs = defaults | overrides
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                CakeNVFP4AttentionImpl(**kwargs)

    def test_forward_uses_nhd_and_reuses_caller_owned_buffers(self):
        impl = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        query = Mock()
        query.shape = (1, 4800, 40, 128)
        query.ndim = 4
        query.device = SimpleNamespace(type="cuda", index=0)
        query.dtype = torch.bfloat16
        query.is_contiguous.return_value = True
        key = Mock()
        key.shape = query.shape
        key.ndim = query.ndim
        key.device = query.device
        key.dtype = query.dtype
        key.is_contiguous.return_value = True
        value = Mock()
        value.shape = query.shape
        value.ndim = query.ndim
        value.device = query.device
        value.dtype = query.dtype
        value.is_contiguous.return_value = True

        packed_type = namedtuple(
            "Packed",
            (
                "q_fp4",
                "k_fp4",
                "v_fp4_t",
                "q_scale",
                "k_scale",
                "v_scale_lo",
                "v_scale_hi",
            ),
        )
        packed = packed_type(*(object() for _ in packed_type._fields))
        allocated_workspace = SimpleNamespace(packed=packed)
        output = object()
        correction_workspace = object()
        qk_correction = object()
        allocate = Mock(return_value=allocated_workspace)
        run = Mock(side_effect=lambda *_args, **kwargs: kwargs["out"])
        quantize_v = Mock()
        quantize_module = SimpleNamespace(quantize_v=quantize_v)
        get_module = Mock(return_value=quantize_module)
        target_for_device = Mock(return_value="sm100a")
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.allocate_cake_nvfp4_attention_workspace = allocate
        fake_flashinfer.cake_nvfp4_attention_fwd = run
        fake_cake_module = ModuleType("flashinfer.cake_nvfp4_attention")
        fake_cake_module._get_module = get_module
        fake_cake_module._target_for_device = target_for_device
        with (
            patch.dict(_SHARED_SCRATCH, {}, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": fake_flashinfer,
                    "flashinfer.cake_nvfp4_attention": fake_cake_module,
                },
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "cake_nvfp4.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=17),
            ),
            patch.object(
                impl,
                "_allocate_correction_workspace",
                return_value=correction_workspace,
            ) as allocate_correction,
            patch.object(
                impl,
                "_prepare_qk_correction",
                return_value=qk_correction,
            ) as prepare_correction,
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
        allocate.assert_called_once_with(query, qkv_layout="NHD")
        allocate_correction.assert_called_once_with(query)
        self.assertEqual(prepare_correction.call_count, 2)
        prepare_correction.assert_called_with(
            query, key, packed, correction_workspace
        )
        empty_like.assert_called_once_with(query, dtype=torch.bfloat16)
        self.assertEqual(quantize_v.call_count, 2)
        quantize_v.assert_called_with(
            value,
            packed.v_fp4_t,
            packed.v_scale_lo,
            packed.v_scale_hi,
            0,
        )
        self.assertEqual(get_module.call_count, 2)
        target_for_device.assert_called_with(value.device)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[:7], tuple(packed))
            self.assertEqual(call.args[7], 4800)
            self.assertEqual(call.kwargs["qkv_layout"], "NHD")
            self.assertIs(call.kwargs["qk_correction"], qk_correction)
            self.assertIs(call.kwargs["out"], output)

    def test_scratch_is_shared_across_layers_on_the_same_stream(self):
        first = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        second = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        query = Mock()
        query.shape = (1, 4800, 40, 128)
        query.device = SimpleNamespace(type="cuda", index=0)
        query.dtype = torch.bfloat16
        packed = object()
        allocated_workspace = SimpleNamespace(packed=packed)
        correction = object()
        allocate = Mock(return_value=allocated_workspace)
        fake_flashinfer = SimpleNamespace(
            allocate_cake_nvfp4_attention_workspace=allocate,
        )

        with (
            patch.dict(_SHARED_SCRATCH, {}, clear=True),
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "cake_nvfp4.torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=23),
            ),
            patch.object(
                CakeNVFP4AttentionImpl,
                "_allocate_correction_workspace",
                return_value=correction,
            ) as allocate_correction,
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "cake_nvfp4.torch.empty_like",
                side_effect=(object(), object()),
            ),
        ):
            first_workspace, first_output, first_correction = (
                first._get_workspace_and_output(query)
            )
            second_workspace, second_output, second_correction = (
                second._get_workspace_and_output(query)
            )

        allocate.assert_called_once_with(query, qkv_layout="NHD")
        allocate_correction.assert_called_once_with(query)
        self.assertIs(first_workspace, second_workspace)
        self.assertIs(first_correction, second_correction)
        self.assertIsNot(first_output, second_output)

    def test_forward_rejects_unqualified_batch(self):
        impl = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        query = torch.empty((2, 512, 40, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            impl.forward(query, query, query, None)

    def test_wan_projection_path_rejects_rank_before_reading_sequence(self):
        impl = CakeNVFP4AttentionImpl(
            num_heads=40,
            num_kv_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        projection = torch.empty((5120,), dtype=torch.bfloat16)
        weight = torch.empty((5120,), dtype=torch.bfloat16)
        cos_sin_cache = torch.empty((1, 128), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, r"\[1, seq_len, 5120\]"):
            impl.forward_wan_projections(
                projection,
                projection,
                projection,
                weight,
                weight,
                cos_sin_cache,
                eps=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
