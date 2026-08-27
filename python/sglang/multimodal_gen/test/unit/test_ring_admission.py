# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend


class TestRingAdmission(unittest.TestCase):
    def test_default_is_not_ring_capable(self):
        self.assertFalse(AttentionBackend.supports_ring_rotation())
        self.assertFalse(SDPABackend.supports_ring_rotation())

    def test_flash_attention_declares_ring_support(self):
        self.assertTrue(FlashAttentionBackend.supports_ring_rotation())


if __name__ == "__main__":
    unittest.main()
