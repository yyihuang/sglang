import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
    flashinfer_gdn_prefill_default,
)
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
    maybe_build_flashinfer_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _publish(testcase, **fields):
    """Install a published config for one case and restore on its cleanup --
    a failed case must not leave a partial publish for a later file in a
    monolithic local run."""
    from sglang.srt.runtime_context import get_context, get_server_args

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)
    return get_server_args()


def make_runner(
    testcase,
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    **arg_overrides,
):
    # The policy reads the published bags, so the fixture publishes the
    # configuration under test.
    fields = dict(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
    )
    fields.update(arg_overrides)
    args = _publish(testcase, **fields)

    return SimpleNamespace(
        server_args=args,
        model_config=SimpleNamespace(),
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestFlashInferGDNPrefillBackendPolicy(unittest.TestCase):
    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(
                gdn_backend,
                "hybrid_gdn_config",
                return_value=runner.hybrid_gdn_config,
            ),
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            return flashinfer_gdn_prefill_default(runner)

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner(self)), "flashinfer")

    def test_selects_flashinfer_for_radix_cache_strategies(self):
        for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                runner = make_runner(
                    self,
                    uses_mamba_radix_cache=True,
                    mamba_radix_cache_strategy=strategy,
                )
                self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_declines_when_the_prefill_backend_is_explicit(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                runner = make_runner(self, linear_attn_prefill_backend=backend)
                self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(self, **runner_args), **hardware)
                )

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner(self)
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("page_major_kv", {"enable_page_major_kv_layout": True}),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(self, **runner_args)))

    def test_builds_compact_checkpoint_plan_for_packed_sequences(self):
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(4),
            track_ssm_h_dst=torch.empty(4),
            checkpoint_extend_seq_lens_cpu=torch.tensor(
                [63, 64, 65, 127, 128, 129]
            ),
            checkpoint_track_mask_cpu=torch.tensor(
                [False, True, True, True, True, True]
            ),
            # 65 on the 128-token sequence represents an interior S64
            # boundary encoded as S64 + 1 by the scheduler.
            checkpoint_relative_track_lens_cpu=torch.tensor(
                [63, 64, 65, 127, 65, 129]
            ),
        )

        # The chunk size is a derived config member; seed its private cache on a
        # published config rather than patching an import binding.
        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)
        maybe_build_flashinfer_checkpoint_plan(metadata, "cpu")

        torch.testing.assert_close(
            metadata.state_checkpoint_cu_starts,
            torch.tensor([0, 0, 1, 2, 3, 5, 7], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.cake_state_checkpoint_cu_starts,
            torch.tensor([0, 0, 1, 2, 3, 5, 7], dtype=torch.int32),
        )
        torch.testing.assert_close(metadata.track_ssm_h_src, torch.tensor([1, 2, 3, 6]))
        self.assertEqual(metadata.num_state_checkpoints, 7)
        self.assertEqual(metadata.state_checkpoint_every_n_tokens, 64)

    def test_tracked_state_planning_exports_the_same_cpu_intermediates(self):
        backend = object.__new__(Mamba2AttnBackend)
        backend.device = "cpu"
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([64, 65], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0, 64], dtype=torch.int32),
            mamba_track_mask=torch.tensor([True, True]),
            mamba_track_indices=torch.tensor([7, 8], dtype=torch.int64),
            mamba_track_seqlens=torch.tensor([64, 65], dtype=torch.int64),
        )

        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)
        outputs = backend._init_track_ssm_indices(
            torch.tensor([3, 4], dtype=torch.int64), forward_batch
        )

        torch.testing.assert_close(outputs[4], forward_batch.extend_seq_lens)
        torch.testing.assert_close(outputs[5], forward_batch.mamba_track_mask)
        torch.testing.assert_close(outputs[6], torch.tensor([64, 1]))
        self.assertEqual(outputs[4].device.type, "cpu")
        self.assertEqual(outputs[5].device.type, "cpu")
        self.assertEqual(outputs[6].device.type, "cpu")

    def test_checkpoint_plan_fails_closed_without_complete_cpu_metadata(self):
        base = dict(
            checkpoint_extend_seq_lens_cpu=torch.tensor([64]),
            checkpoint_track_mask_cpu=torch.tensor([True]),
            checkpoint_relative_track_lens_cpu=torch.tensor([64]),
        )

        for missing in base:
            with self.subTest(missing=missing):
                fields = dict(base)
                fields[missing] = None
                metadata = SimpleNamespace(
                    track_ssm_h_src=torch.empty(1),
                    track_ssm_h_dst=torch.empty(1),
                    **fields,
                )
                with self.assertRaisesRegex(RuntimeError, missing):
                    maybe_build_flashinfer_checkpoint_plan(metadata, "cpu")

    def test_checkpoint_plan_fails_closed_on_cpu_metadata_length_mismatch(self):
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(1),
            track_ssm_h_dst=torch.empty(1),
            checkpoint_extend_seq_lens_cpu=torch.tensor([64, 64]),
            checkpoint_track_mask_cpu=torch.tensor([True]),
            checkpoint_relative_track_lens_cpu=torch.tensor([64, 64]),
        )

        with self.assertRaisesRegex(ValueError, "checkpoint_track_mask_cpu"):
            maybe_build_flashinfer_checkpoint_plan(metadata, "cpu")

    def test_mixed_prefill_decode_uses_cpu_prefix_mirror_without_tensor_sync(self):
        checkpoint_extend_seq_lens_cpu = torch.tensor([2, 1])
        checkpoint_track_mask_cpu = torch.tensor([False, False])
        checkpoint_relative_track_lens_cpu = torch.tensor([2, 1])
        forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            checkpoint_extend_seq_lens_cpu=checkpoint_extend_seq_lens_cpu,
            checkpoint_track_mask_cpu=checkpoint_track_mask_cpu,
            checkpoint_relative_track_lens_cpu=checkpoint_relative_track_lens_cpu,
        )
        forward_batch = SimpleNamespace(
            extend_num_tokens=3,
            extend_seq_lens_cpu=[2, 1],
            extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=[0, 0],
            extend_prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([2, 1, 9], dtype=torch.int32),
            _original_batch_size=3,
            spec_info=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with patch.object(
            torch,
            "any",
            side_effect=AssertionError("CPU mirror path must not call torch.any"),
        ):
            metadata = Mamba2Metadata.prepare_mixed(
                forward_metadata, chunk_size=64, forward_batch=forward_batch
            )

        self.assertFalse(metadata.mixed_metadata.prep_initial_states)
        self.assertEqual(metadata.num_prefills, 2)
        self.assertEqual(metadata.num_decodes, 1)
        self.assertIs(
            metadata.checkpoint_extend_seq_lens_cpu,
            checkpoint_extend_seq_lens_cpu,
        )
        self.assertIs(metadata.checkpoint_track_mask_cpu, checkpoint_track_mask_cpu)
        self.assertIs(
            metadata.checkpoint_relative_track_lens_cpu,
            checkpoint_relative_track_lens_cpu,
        )

    def test_ordinary_prefill_uses_cpu_prefix_mirror_without_tensor_sync(self):
        forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        forward_batch = SimpleNamespace(
            extend_num_tokens=3,
            extend_seq_lens_cpu=[2, 1],
            extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=[0, 0],
            extend_prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            spec_info=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with patch.object(
            torch,
            "any",
            side_effect=AssertionError("CPU mirror path must not call torch.any"),
        ):
            metadata = Mamba2Metadata.prepare_mixed(
                forward_metadata, chunk_size=64, forward_batch=forward_batch
            )

        self.assertFalse(metadata.mixed_metadata.prep_initial_states)
        self.assertEqual(metadata.num_prefills, 2)
        self.assertEqual(metadata.num_decodes, 0)

    def test_mamba_prefill_missing_cpu_prefix_mirror_fails_closed_without_sync(self):
        forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([0], dtype=torch.int64),
        )
        forward_batch = SimpleNamespace(
            extend_num_tokens=2,
            extend_seq_lens_cpu=[2],
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_prefix_lens_cpu=None,
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            spec_info=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with (
            patch.object(
                torch,
                "any",
                side_effect=AssertionError("fail-closed path must not call torch.any"),
            ),
            self.assertRaisesRegex(RuntimeError, "extend_prefix_lens_cpu"),
        ):
            Mamba2Metadata.prepare_mixed(
                forward_metadata, chunk_size=64, forward_batch=forward_batch
            )

    def test_mamba_mixed_metadata_fails_closed_on_cpu_mirror_length_mismatch(self):
        forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        forward_batch = SimpleNamespace(
            extend_num_tokens=3,
            extend_seq_lens_cpu=[2, 1],
            extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=[0],
            extend_prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            spec_info=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with self.assertRaisesRegex(ValueError, "expected 2 prefix lengths"):
            Mamba2Metadata.prepare_mixed(
                forward_metadata, chunk_size=64, forward_batch=forward_batch
            )

    def test_decode_tracking_without_h_source_skips_checkpoint_plan(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
        metadata = SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_src=None)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7]),
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base):
            backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(metadata.conv_states_mask_indices, torch.tensor([7]))

    def test_target_verify_without_track_mask_skips_checkpoint_planning(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
        metadata = SimpleNamespace(has_mamba_track_mask=False)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: True)
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with (
            patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "maybe_build_flashinfer_checkpoint_plan"
            ) as checkpoint_plan,
        ):
            backend.init_forward_metadata(forward_batch)

        checkpoint_plan.assert_not_called()

    def test_indexed_prefill_keeps_original_ssm_pool_when_conv_needs_packing(self):
        backend = object.__new__(GDNAttnBackend)
        cache_indices = torch.tensor([3], dtype=torch.int32)
        conv_states = torch.empty(8, 6, 3, dtype=torch.bfloat16)[::2]
        ssm_states = torch.empty(8, 1, 2, 2, dtype=torch.bfloat16)[::2]
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: SimpleNamespace(
                conv=[conv_states], temporal=ssm_states
            )
        )
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            mamba_cache_indices=cache_indices,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
            has_mamba_track_mask=False,
            state_checkpoint_cu_starts=None,
            cake_state_checkpoint_cu_starts=None,
            num_state_checkpoints=0,
            state_checkpoint_every_n_tokens=0,
        )
        dispatcher = SimpleNamespace(
            extend_supports_indexed_state_pool=True,
            extend=MagicMock(
                return_value=(
                    torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                    None,
                    None,
                )
            ),
        )
        backend.kernel_dispatcher = dispatcher
        layer = SimpleNamespace(
            layer_id=7,
            conv_weights=sentinel.conv_weights,
            bias=sentinel.bias,
            activation=sentinel.activation,
            q_dim=2,
            k_dim=2,
            v_dim=2,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
            A_log=torch.empty(1),
            dt_bias=torch.empty(1),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[2],
        )
        causal_conv = MagicMock(side_effect=lambda value, *_args, **_kwargs: value)

        with (
            patch.object(gdn_backend, "is_cpu", return_value=False),
            patch.object(gdn_backend, "causal_conv1d_fn", causal_conv),
            patch.object(
                gdn_backend,
                "fused_qkv_split_gdn_prefill",
                return_value=(
                    torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                    torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                    torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                ),
            ),
            patch.object(
                gdn_backend,
                "fused_gdn_gating",
                return_value=(torch.empty(1, 2, 1), torch.empty(1, 2, 1)),
            ),
        ):
            backend.forward_extend(
                layer,
                forward_batch,
                torch.empty(2, 6, dtype=torch.bfloat16),
                torch.empty(2, 1),
                torch.empty(2, 1),
            )

        conv_kwargs = causal_conv.call_args.kwargs
        self.assertTrue(conv_kwargs["conv_states"].is_contiguous())
        self.assertNotEqual(
            conv_kwargs["conv_states"].data_ptr(), conv_states.data_ptr()
        )
        torch.testing.assert_close(
            conv_kwargs["cache_indices"], torch.tensor([0], dtype=torch.int32)
        )
        extend_kwargs = dispatcher.extend.call_args.kwargs
        self.assertEqual(extend_kwargs["ssm_states"].data_ptr(), ssm_states.data_ptr())
        self.assertIs(extend_kwargs["cache_indices"], cache_indices)

    def test_flashinfer_fallback_gathers_noncontiguous_state_once_and_writes_back(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class CountStateGather(TorchDispatchMode):
            def __init__(self):
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.index.Tensor:
                    self.count += 1
                return func(*args, **(kwargs or {}))

        kernel = object.__new__(FlashInferGDNKernel)
        kernel.use_state_pool = True
        kernel._try_cake_prefill = MagicMock(return_value=None)
        output = torch.empty(2, 1, 2, dtype=torch.bfloat16)

        def flashinfer_prefill(**kwargs):
            kwargs["output_state"].copy_(kwargs["initial_state"] + 1)
            return output, kwargs["output_state"]

        kernel._prefill_fn = MagicMock(side_effect=flashinfer_prefill)
        state_storage = torch.zeros(8, 1, 2, 2, dtype=torch.bfloat16)
        state_pool = state_storage[::2]
        state_pool[3].fill_(2)
        state_indices = torch.tensor([3], dtype=torch.int32)
        counter = CountStateGather()

        with (
            patch(
                "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
                side_effect=lambda value: value,
            ),
            counter,
        ):
            kernel.extend(
                q=torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                k=torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                v=torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
                g=torch.empty(1, 2, 1, dtype=torch.float32),
                beta=torch.empty(1, 2, 1, dtype=torch.float32),
                ssm_states=state_pool,
                cache_indices=state_indices,
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
                seq_lens_cpu=[2],
                layer_id=7,
            )

        self.assertEqual(counter.count, 1)
        torch.testing.assert_close(
            state_pool[3], torch.full_like(state_pool[3], 3)
        )

    def test_flashinfer_indexed_prefill_pool_capability_is_sm100_family_only(self):
        kernels = (True, sentinel.prefill, sentinel.mtp, sentinel.decode, None)

        for capability, expected in (((10, 3), True), ((12, 0), False)):
            with (
                self.subTest(capability=capability),
                patch(
                    "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                    "_get_flashinfer_gdn_kernels",
                    return_value=kernels,
                ),
                patch(
                    "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                    "_get_cake_gdn_decode_api",
                    return_value=None,
                ),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ),
            ):
                kernel = FlashInferGDNKernel()

            self.assertIs(kernel.supports_indexed_prefill_state_pool, expected)

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(supports_target_verify=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()

    def test_helion_backend_reports_kda_only(self):
        cases = (
            (LinearAttnKernelBackend.HELION, LinearAttnKernelBackend.TRITON),
            (LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.HELION),
        )
        for decode_backend, prefill_backend in cases:
            with self.subTest(
                decode_backend=decode_backend,
                prefill_backend=prefill_backend,
            ):
                with self.assertRaisesRegex(ValueError, "supports KDA only"):
                    GDNKernelDispatcher(decode_backend, prefill_backend)


if __name__ == "__main__":
    unittest.main()
