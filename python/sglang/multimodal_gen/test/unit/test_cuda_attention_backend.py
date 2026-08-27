# SPDX-License-Identifier: Apache-2.0

import sys
import types
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.platforms.cuda import (
    _WanHybridAttentionBackendResolver,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability


class _FakeCudaPlatform:
    capability = DeviceCapability(10, 0)

    @classmethod
    def get_device_capability(cls):
        return cls.capability


class TestWanHybridCudaAttentionBackend(unittest.TestCase):
    def _flashinfer(self, *, available=True, omit=None):
        module = types.ModuleType("flashinfer")
        exports = {
            "WanHybridAttentionWorkspace": object(),
            "wan_hybrid_attention": object(),
            "is_wan_hybrid_attention_available": lambda: available,
        }
        for name, value in exports.items():
            if name != omit:
                setattr(module, name, value)
        return module

    def test_accepts_sm100_and_sm103(self):
        expected = "sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid.WanHybridAttentionBackend"
        with patch.dict(sys.modules, {"flashinfer": self._flashinfer()}):
            for capability in (DeviceCapability(10, 0), DeviceCapability(10, 3)):
                _FakeCudaPlatform.capability = capability
                self.assertEqual(
                    _WanHybridAttentionBackendResolver.resolve(_FakeCudaPlatform),
                    expected,
                )

    def test_rejects_other_architectures(self):
        _FakeCudaPlatform.capability = DeviceCapability(12, 0)
        with self.assertRaisesRegex(ValueError, "10.0 or 10.3"):
            _WanHybridAttentionBackendResolver.resolve(_FakeCudaPlatform)

    def test_fails_closed_without_public_implementation(self):
        _FakeCudaPlatform.capability = DeviceCapability(10, 0)
        with patch.dict(
            sys.modules, {"flashinfer": self._flashinfer(available=False)}
        ), self.assertRaisesRegex(RuntimeError, "installed FlashInfer"):
            _WanHybridAttentionBackendResolver.resolve(_FakeCudaPlatform)

    def test_fails_closed_without_public_exports(self):
        _FakeCudaPlatform.capability = DeviceCapability(10, 0)
        for missing in (
            "WanHybridAttentionWorkspace",
            "wan_hybrid_attention",
            "is_wan_hybrid_attention_available",
        ):
            with patch.dict(
                sys.modules, {"flashinfer": self._flashinfer(omit=missing)}
            ), self.assertRaisesRegex(ImportError, "public wan_hybrid"):
                _WanHybridAttentionBackendResolver.resolve(_FakeCudaPlatform)


if __name__ == "__main__":
    unittest.main()
