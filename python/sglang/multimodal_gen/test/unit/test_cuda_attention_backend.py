import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms.cuda import (
    CudaPlatformBase,
    _WanHybridAttentionBackendResolver,
    _SageAttentionBackendResolver,
)
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
)

SDPA_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)


class FakeCudaPlatform(CudaPlatformBase):
    is_sm120_device = False
    is_blackwell_device = False
    is_hopper_device = False
    supports_flash_attention = True

    @classmethod
    def is_sm120(cls):
        return cls.is_sm120_device

    @classmethod
    def is_blackwell(cls):
        return cls.is_blackwell_device

    @classmethod
    def is_hopper(cls):
        return cls.is_hopper_device

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls.supports_flash_attention


class TestCudaAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        FakeCudaPlatform.is_sm120_device = False
        FakeCudaPlatform.is_blackwell_device = False
        FakeCudaPlatform.is_hopper_device = False
        FakeCudaPlatform.supports_flash_attention = True
        _cached_get_attn_backend.cache_clear()

    def resolve(
        self,
        selected_backend: AttentionBackendEnum | None,
        dtype: torch.dtype = torch.float16,
    ) -> str:
        return FakeCudaPlatform.get_attn_backend_cls_str(
            selected_backend=selected_backend,
            head_size=128,
            dtype=dtype,
        )

    def test_direct_torch_sdpa_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.TORCH_SDPA), SDPA_BACKEND_CLS_STR
        )

    def test_direct_aiter_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.AITER),
            "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend",
        )

    def test_default_backend_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_requested_flash_attention_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(AttentionBackendEnum.FA), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_for_non_flash_attention_dtype(self):
        self.assertEqual(self.resolve(None, torch.float32), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_without_flash_attention_capability(self):
        FakeCudaPlatform.supports_flash_attention = False

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_blackwell_falls_back_when_flash_attention_prepare_fails(self):
        FakeCudaPlatform.is_blackwell_device = True

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=False,
        ) as prepare_flash_attention:
            self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

        prepare_flash_attention.assert_called_once_with()

    def test_default_backend_prefers_dynamic_cudnn_sdpa_on_blackwell(self):
        FakeCudaPlatform.is_blackwell_device = True

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=True,
        ):
            self.assertEqual(
                self.resolve(None),
                "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.DynamicCudnnSDPABackend",
            )

    def test_invalid_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid attention backend"):
            self.resolve(AttentionBackendEnum.AITER_SAGE)

    def test_wan_hybrid_resolver_accepts_sm100_and_sm103(self):
        flashinfer = types.ModuleType("flashinfer")
        flashinfer.WanHybridAttentionWorkspace = object()
        flashinfer.wan_hybrid_attention = object()
        flashinfer.is_wan_hybrid_attention_available = lambda: True
        expected = "sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid.WanHybridAttentionBackend"
        with patch.dict(sys.modules, {"flashinfer": flashinfer}):
            for capability in (DeviceCapability(10, 0), DeviceCapability(10, 3)):
                with self.subTest(capability=capability), patch.object(
                    FakeCudaPlatform,
                    "get_device_capability",
                    return_value=capability,
                ):
                    self.assertEqual(
                        _WanHybridAttentionBackendResolver.resolve(FakeCudaPlatform),
                        expected,
                    )

    def test_wan_hybrid_resolver_fails_closed_without_public_impl(self):
        flashinfer = types.ModuleType("flashinfer")
        flashinfer.WanHybridAttentionWorkspace = object()
        flashinfer.wan_hybrid_attention = object()
        flashinfer.is_wan_hybrid_attention_available = lambda: False
        with patch.dict(sys.modules, {"flashinfer": flashinfer}), patch.object(
            FakeCudaPlatform,
            "get_device_capability",
            return_value=DeviceCapability(10, 0),
        ), self.assertRaisesRegex(RuntimeError, "installed FlashInfer"):
            _WanHybridAttentionBackendResolver.resolve(FakeCudaPlatform)

    def test_wan_hybrid_resolver_rejects_other_architectures(self):
        with patch.object(
            FakeCudaPlatform,
            "get_device_capability",
            return_value=DeviceCapability(12, 0),
        ), self.assertRaisesRegex(ValueError, "10.0 or 10.3"):
            _WanHybridAttentionBackendResolver.resolve(FakeCudaPlatform)

    def test_hopper_sage_attention_without_sm90_fix_falls_back(self):
        FakeCudaPlatform.is_hopper_device = True
        sageattention = types.ModuleType("sageattention")
        sageattention.__path__ = []
        sageattention.sageattn = object()
        sm90_compile = types.ModuleType("sageattention.sm90_compile")

        with patch.dict(
            sys.modules,
            {
                "sageattention": sageattention,
                "sageattention.sm90_compile": sm90_compile,
            },
        ):
            self.assertEqual(
                _SageAttentionBackendResolver.resolve(FakeCudaPlatform),
                AttentionBackendEnum.FA,
            )

    def test_explicit_backend_rejected_by_a_model_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            _cached_get_attn_backend(
                128,
                torch.float16,
                (AttentionBackendEnum.FA,),
                AttentionBackendEnum.SAGE_ATTN,
            )


if __name__ == "__main__":
    unittest.main()
