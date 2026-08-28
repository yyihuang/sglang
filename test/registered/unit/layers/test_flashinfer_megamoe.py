import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.mega_moe import (
    build_flashinfer_megamoe_experts_weights,
)


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _WeightPack:
    def __init__(self, w13, w2):
        self.w13 = w13
        self.w2 = w2


class _FakeMoEEpLayer(torch.nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.calls.append(kwargs)


class _Experts(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 4, 8, dtype=dtype), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 8, 2, dtype=dtype), requires_grad=False
        )
        self.with_bias = False
        self.num_fused_shared_experts = 0
        self.num_local_experts = 2
        self.num_experts = 4
        self.moe_ep_size = 2
        self.moe_ep_rank = 1
        self.moe_tp_size = 1
        self.hidden_size = 8
        self.intermediate_size_per_partition = 2
        self.top_k = 2
        self.moe_runner_config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            swiglu_limit=None,
        )


def _fake_flashinfer_modules():
    moe_ep = types.ModuleType("flashinfer.moe_ep")
    moe_ep.BootstrapConfig = _Config
    moe_ep.FleetParams = _Config
    moe_ep.MegaConfig = _Config
    moe_ep.MoEEpLayer = _FakeMoEEpLayer
    moe_ep.MoEWeightPack = _WeightPack
    moe_ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig = _Config
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.moe_ep = moe_ep
    return {"flashinfer": flashinfer, "flashinfer.moe_ep": moe_ep}


class TestFlashInferMegaMoEWeightOwnership(unittest.TestCase):
    def setUp(self):
        _FakeMoEEpLayer.calls.clear()

    def test_public_api_build_transfers_parameter_ownership(self):
        experts = _Experts()
        ep_group = object()
        runtime = SimpleNamespace(
            moe=SimpleNamespace(flashinfer_megamoe_max_tokens_per_rank=16)
        )
        with (
            patch.dict(sys.modules, _fake_flashinfer_modules()),
            patch(
                "sglang.srt.layers.moe.mega_moe.get_exec", return_value=runtime
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=ep_group),
            ),
            patch("torch.cuda.current_device", return_value=1),
            patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
        ):
            build_flashinfer_megamoe_experts_weights(experts)

        self.assertTrue(experts._flashinfer_megamoe_weights_built)
        self.assertFalse(experts._flashinfer_megamoe_warmed_up)
        self.assertIsNone(experts.w13_weight)
        self.assertIsNone(experts.w2_weight)
        self.assertIsInstance(experts.flashinfer_megamoe_layer, _FakeMoEEpLayer)
        self.assertEqual(len(_FakeMoEEpLayer.calls), 1)
        call = _FakeMoEEpLayer.calls[0]
        self.assertIs(call["bootstrap"].kwargs["process_group"], ep_group)
        self.assertEqual(call["bootstrap"].kwargs["world_size"], 2)
        self.assertEqual(call["fleet_params"].kwargs["max_tokens_per_rank"], 16)
        self.assertEqual(
            call["backend"].kwargs["megakernel"].kwargs["intermediate_size"], 2
        )

    def test_invalid_weight_dtype_keeps_original_parameters(self):
        experts = _Experts(dtype=torch.float32)
        with patch.dict(sys.modules, _fake_flashinfer_modules()):
            with self.assertRaisesRegex(ValueError, "canonical BF16"):
                build_flashinfer_megamoe_experts_weights(experts)

        self.assertIsNotNone(experts.w13_weight)
        self.assertIsNotNone(experts.w2_weight)
        self.assertFalse(hasattr(experts, "flashinfer_megamoe_layer"))

    def test_unquantized_post_load_hook_delegates_to_owner_builder(self):
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
        )

        experts = _Experts()
        backend = SimpleNamespace(is_flashinfer_megamoe=lambda: True)
        with (
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_a2a_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.moe.mega_moe."
                "build_flashinfer_megamoe_experts_weights"
            ) as build,
        ):
            UnquantizedFusedMoEMethod.process_weights_after_loading(
                object(), experts
            )

        build.assert_called_once_with(experts)


if __name__ == "__main__":
    unittest.main()
