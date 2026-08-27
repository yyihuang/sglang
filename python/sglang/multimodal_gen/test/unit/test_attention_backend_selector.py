# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class _FakeFABackend:
    @classmethod
    def get_enum(cls):
        return AttentionBackendEnum.FA


class _FakeWanHybridBackend:
    @classmethod
    def get_enum(cls):
        return AttentionBackendEnum.WAN_HYBRID


class _FakePlatform:
    selected_backend = None

    @classmethod
    def get_attn_backend_cls_str(cls, selected_backend, _head_size, _dtype):
        cls.selected_backend = selected_backend
        if selected_backend is AttentionBackendEnum.WAN_HYBRID:
            return "fake.WanHybrid"
        return "fake.FA"


class TestWanHybridAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        _cached_get_attn_backend.cache_clear()
        _FakePlatform.selected_backend = None

    def _resolve(self, *, is_cross_attention):
        backends = {
            "fake.FA": _FakeFABackend,
            "fake.WanHybrid": _FakeWanHybridBackend,
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform",
                _FakePlatform,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.resolve_obj_by_qualname",
                side_effect=backends.__getitem__,
            ),
        ):
            return get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.WAN_HYBRID,
                },
                selected_attention_backend=AttentionBackendEnum.WAN_HYBRID,
                is_cross_attention=is_cross_attention,
            )

    def test_wan_hybrid_falls_back_for_cross_attention(self):
        self.assertIs(self._resolve(is_cross_attention=True), _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_wan_hybrid_remains_strict_for_self_attention(self):
        self.assertIs(
            self._resolve(is_cross_attention=False), _FakeWanHybridBackend
        )


if __name__ == "__main__":
    unittest.main()
