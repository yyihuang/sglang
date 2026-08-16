# SPDX-License-Identifier: Apache-2.0

import unittest

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
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, message
            ):
                CakeNVFP4AttentionImpl(**kwargs)


if __name__ == "__main__":
    unittest.main()
