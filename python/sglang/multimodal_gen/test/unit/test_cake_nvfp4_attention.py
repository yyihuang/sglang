# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.cake_nvfp4 import (
    CakeNVFP4AttentionBackend,
    CakeNVFP4AttentionImpl,
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

        workspace = object()
        output = object()
        correction_workspace = object()
        centered_query = Mock()
        centered_key = Mock()
        qk_correction = object()
        allocate = Mock(return_value=workspace)
        run = Mock(side_effect=lambda *_args, **kwargs: kwargs["out"])
        fake_flashinfer = SimpleNamespace(
            allocate_cake_nvfp4_attention_workspace=allocate,
            nvfp4_attention=run,
        )
        with (
            patch.dict("sys.modules", {"flashinfer": fake_flashinfer}),
            patch.object(
                impl,
                "_allocate_correction_workspace",
                return_value=correction_workspace,
            ) as allocate_correction,
            patch.object(
                impl,
                "_prepare_qk_correction",
                return_value=(centered_query, centered_key, qk_correction),
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
        prepare_correction.assert_called_with(query, key, correction_workspace)
        empty_like.assert_called_once_with(query, dtype=torch.bfloat16)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["backend"], "cake")
            self.assertEqual(call.kwargs["qkv_layout"], "NHD")
            self.assertIs(call.kwargs["workspace"], workspace)
            self.assertIs(call.args[0], centered_query)
            self.assertIs(call.args[1], centered_key)
            self.assertIs(call.kwargs["qk_correction"], qk_correction)
            self.assertIs(call.kwargs["out"], output)

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


if __name__ == "__main__":
    unittest.main()
