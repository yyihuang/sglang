import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.mega_moe import (
    _flashinfer_megamoe_chunk_ranges,
    _run_flashinfer_mega_routed,
    build_flashinfer_megamoe_experts_weights,
)


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _WeightPack:
    def __init__(self, w13, w2):
        self.w13 = w13
        self.w2 = w2


class _TensorBundle:
    def __init__(self, *, hidden_states, topk_ids, topk_weights):
        self.hidden_states = hidden_states
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights


class _FakeMoEEpLayer(torch.nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.calls.append(kwargs)


class _EpAlgorithm:
    LOW_LATENCY = object()


class _EpLayout:
    RANK_MAJOR = object()


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
        self.num_local_experts = 32
        self.num_experts = 256
        self.moe_ep_size = 8
        self.moe_ep_rank = 1
        self.moe_tp_size = 1
        self.hidden_size = 7168
        self.intermediate_size_per_partition = 2048
        self.top_k = 8
        self.moe_runner_config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            swiglu_limit=None,
        )


def _fake_flashinfer_modules():
    moe_ep = types.ModuleType("flashinfer.moe_ep")
    moe_ep.BootstrapConfig = _Config
    moe_ep.EpAlgorithm = _EpAlgorithm
    moe_ep.EpLayout = _EpLayout
    moe_ep.FleetParams = _Config
    moe_ep.MegaConfig = _Config
    moe_ep.MoEEpLayer = _FakeMoEEpLayer
    moe_ep.MoEWeightPack = _WeightPack
    moe_ep.MoEEpTensors = _TensorBundle
    moe_ep.Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig = _Config
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
        with (
            patch.dict(sys.modules, _fake_flashinfer_modules()),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=ep_group),
            ),
            patch("torch.cuda.current_device", return_value=1),
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
        self.assertEqual(call["bootstrap"].kwargs["world_size"], 8)
        self.assertEqual(call["bootstrap"].kwargs["stream"], 0)
        self.assertEqual(call["fleet_params"].kwargs["max_tokens_per_rank"], 128)
        self.assertEqual(call["fleet_params"].kwargs["num_experts"], 256)
        self.assertEqual(call["fleet_params"].kwargs["token_hidden_size"], 7168)
        self.assertEqual(call["fleet_params"].kwargs["dtype_bytes"], 2)
        self.assertIs(
            call["fleet_params"].kwargs["algorithm"], _EpAlgorithm.LOW_LATENCY
        )
        self.assertIs(call["fleet_params"].kwargs["layout"], _EpLayout.RANK_MAJOR)
        self.assertEqual(
            call["backend"].kwargs["megakernel"].kwargs["intermediate_size"],
            2048,
        )
        self.assertEqual(call["backend"].kwargs["megakernel"].kwargs["top_k"], 8)
        self.assertFalse(hasattr(experts, "_flashinfer_megamoe_hidden_buffer"))
        self.assertFalse(hasattr(experts, "_flashinfer_megamoe_topk_ids_buffer"))
        self.assertFalse(hasattr(experts, "_flashinfer_megamoe_topk_weights_buffer"))

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


class TestFlashInferMegaMoEActiveChunks(unittest.TestCase):
    def test_chunk_ranges_cover_fixed_boundaries(self):
        expected = {
            0: (),
            1: ((0, 1),),
            127: ((0, 127),),
            128: ((0, 128),),
            129: ((0, 128), (128, 129)),
            256: ((0, 128), (128, 256)),
            257: ((0, 128), (128, 256), (256, 257)),
        }
        for num_tokens, ranges in expected.items():
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(_flashinfer_megamoe_chunk_ranges(num_tokens), ranges)

    def test_routed_forward_passes_actual_chunk_shapes(self):
        calls = []

        class Layer:
            def __call__(self, tensors):
                calls.append(
                    (
                        tuple(tensors.hidden_states.shape),
                        tuple(tensors.topk_ids.shape),
                        tuple(tensors.topk_weights.shape),
                        tensors.topk_ids.dtype,
                    )
                )
                return tensors.hidden_states.clone()

        num_tokens = 129
        hidden = (
            torch.arange(num_tokens, dtype=torch.bfloat16)
            .view(num_tokens, 1)
            .expand(num_tokens, 7168)
            .contiguous()
        )
        ids = torch.zeros((num_tokens, 8), dtype=torch.int32)
        weights = torch.full((num_tokens, 8), 0.125, dtype=torch.float32)
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                flashinfer_megamoe_layer=Layer(),
                _flashinfer_megamoe_warmed_up=True,
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            gate=lambda *_args, **_kwargs: object(),
            topk=lambda *_args, **_kwargs: SimpleNamespace(
                topk_ids=ids, topk_weights=weights
            ),
            is_hash=False,
            layer_id=0,
        )
        with (
            patch.dict(sys.modules, _fake_flashinfer_modules()),
            patch(
                "sglang.srt.layers.moe.mega_moe.ExpertLocationDispatchInfo.init_new",
                return_value=object(),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            output = _run_flashinfer_mega_routed(moe, hidden, None, None, num_tokens)

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        torch.testing.assert_close(output, hidden)
        self.assertEqual(
            calls,
            [
                ((128, 7168), (128, 8), (128, 8), torch.int32),
                ((1, 7168), (1, 8), (1, 8), torch.int32),
            ],
        )


if __name__ == "__main__":
    unittest.main()
