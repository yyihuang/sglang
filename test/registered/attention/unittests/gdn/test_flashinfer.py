import unittest
from dataclasses import replace
from unittest import mock

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    _cache_indices,
    _pure_torch_gdn_reference,
    _ssm_states,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
    run_gdn_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

_cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
_sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
_supports_flashinfer_linear_gdn = _sm_major == 9 or (
    _sm_major == 10 and _cuda_major >= 13
)


def _install_t39_padded_state_pool(fixture) -> None:
    """Give the traced fixture the exact four-slot production envelope stride."""

    pool = fixture.runner.req_to_token_pool.mamba_pool
    temporal = pool.mamba_cache.temporal
    assert tuple(temporal.shape) == (1, 4, 8, 128, 128)
    padded = torch.empty_strided(
        temporal.shape,
        (4 * 131185, 131185, 16384, 128, 1),
        dtype=temporal.dtype,
        device=temporal.device,
    )
    padded.copy_(temporal)
    pool.mamba_cache = replace(pool.mamba_cache, temporal=padded)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferGDNBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64

    CASES = make_gdn_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(
                        self,
                        case,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_flashinfer_available()
    and _supports_flashinfer_linear_gdn,
    "FlashInfer linear GDN requires SM90 or SM100/SM103 with CUDA 13+",
)
class TestFlashInferLinearGDNBackendCorrectness(CustomTestCase):
    # FlashInfer's DSL prefill kernels require head size 128 on SM90 and SM100.
    HEAD_DIM = 128
    CHECKPOINT_CASE = GDNAttentionCase(
        name="flashinfer_gdn_prefill_state_checkpoints",
        backend="triton",
        linear_attn_prefill_backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=2,
        num_v_heads=4,
        page_size=16,
        prefix_lens=(0, 64, 128),
        extend_lens=(64, 65, 129),
    )
    CAKE_DECODE_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_decode_b4_t1",
        backend="flashinfer",
        forward_mode=ForwardMode.DECODE,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 7, 10, 13),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_prefill_b5_s64",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 7, 10, 13, 16),
        extend_lens=(64,) * 5,
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_CHECKPOINT_PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_prefill_b7_t421_checkpoints",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 7, 10, 13, 16, 19, 22),
        extend_lens=(52, 93, 15, 107, 72, 61, 21),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_CHECKPOINT_PREFILL_B1_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_prefill_b1_t103_checkpoint",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4,),
        extend_lens=(103,),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_CP_PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_cp_prefill_b1_s128",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4,),
        extend_lens=(128,),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_CP_PREFILL_T39_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_cp_prefill_b1_t39",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4,),
        extend_lens=(39,),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_FP32_DECODE_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_fp32_decode_b1_t1",
        backend="flashinfer",
        forward_mode=ForwardMode.DECODE,
        num_k_heads=16,
        num_v_heads=32,
        page_size=16,
        prefix_lens=(4,),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="triton",
    )
    CAKE_VERIFY_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_verify_b8_t4",
        backend="flashinfer",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 5, 6, 7, 8, 9, 10, 11),
        extend_lens=(4,) * 8,
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_TRACED_VERIFY_CASES = tuple(
        GDNAttentionCase(
            name=f"flashinfer_cake_gdn_tp4_verify_b{batch_size}_t4",
            backend="flashinfer",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_k_heads=4,
            num_v_heads=8,
            page_size=16,
            prefix_lens=tuple(range(4, 4 + batch_size)),
            extend_lens=(4,) * batch_size,
            linear_attn_decode_backend="flashinfer",
            linear_attn_prefill_backend="flashinfer",
        )
        for batch_size in (1, 4, 6, 7)
    )

    def _cake_api_or_skip(self):
        try:
            from flashinfer.jit import cake_gdn_noncp_decode
        except ImportError:
            self.skipTest("public FlashInfer Cake GDN loader is unavailable")
        return cake_gdn_noncp_decode

    def _cake_cp_api_or_skip(self):
        try:
            from flashinfer.jit import cake_gdn_cp_prefill
        except ImportError:
            self.skipTest("public FlashInfer Cake GDN CP loader is unavailable")
        return cake_gdn_cp_prefill

    @staticmethod
    def _cake_decode_test_kernel(cake_api, output):
        kernel = object.__new__(FlashInferGDNKernel)
        kernel._cake_gdn_api = cake_api
        kernel._cake_gdn_arch = cake_api.arch_for_compute_capability(
            *torch.cuda.get_device_capability()
        )
        kernel._cake_gdn_entries = {}
        kernel._cake_gdn_logged_routes = set()
        kernel._cake_fp32_dt_bias = lambda dt_bias, *, layer_id: dt_bias.float()
        kernel._cake_output_buffer = lambda *args, **kwargs: output
        return kernel

    def test_cake_bf16_t2_compact_inputs_select_padded_cache_route(self):
        cake_api = self._cake_api_or_skip()
        batch_size, seq_len, num_q_heads, num_v_heads = 4, 2, 16, 32
        q = torch.randn(
            batch_size,
            seq_len,
            num_q_heads,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch_size,
            seq_len,
            num_v_heads,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        a = torch.randn(
            batch_size,
            seq_len,
            num_v_heads,
            dtype=torch.bfloat16,
            device="cuda",
        )
        b = torch.randn_like(a)
        state = torch.empty_strided(
            (6, num_v_heads, 128, 128),
            (num_v_heads * 128 * 128 + 113, 128 * 128, 128, 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        state_indices = torch.tensor([1, 4, 2, 5], dtype=torch.int32, device="cuda")
        intermediate = torch.empty(
            batch_size,
            4,
            num_v_heads,
            128,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        output = torch.empty_like(v)
        entry = mock.Mock()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_decode_variant",
                wraps=cake_api.select_cake_gdn_decode_variant,
            ) as selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                return_value=entry,
            ),
        ):
            kernel = self._cake_decode_test_kernel(cake_api, output)
            actual = kernel._try_cake_decode(
                q=q,
                k=k,
                v=v,
                state=state,
                state_indices=state_indices,
                A_log=torch.zeros(num_v_heads, dtype=torch.float32, device="cuda"),
                a=a,
                dt_bias=torch.zeros(
                    num_v_heads, dtype=torch.float32, device="cuda"
                ),
                b=b,
                layer_id=0,
                disable_state_update=True,
                intermediate_state=intermediate,
                cache_steps=4,
            )

        self.assertIsNotNone(actual)
        self.assertFalse(selector.call_args.kwargs["strided_inputs"])
        self.assertEqual(
            entry.call_args.args[-3:],
            (batch_size * num_v_heads * 4, 1, 1),
        )

    def test_cake_fp32_t2_strided_inputs_use_inline_tile8_abi(self):
        cake_api = self._cake_api_or_skip()
        batch_size, seq_len, num_q_heads, num_v_heads = 1, 2, 16, 32

        def padded(shape, inner_strides):
            stride0 = shape[1] * inner_strides[0] + 17
            tensor = torch.empty_strided(
                shape,
                (stride0, *inner_strides),
                dtype=torch.bfloat16,
                device="cuda",
            )
            return tensor.normal_()

        q = padded((batch_size, seq_len, num_q_heads, 128), (2048, 128, 1))
        k = padded((batch_size, seq_len, num_q_heads, 128), (2048, 128, 1))
        v = padded((batch_size, seq_len, num_v_heads, 128), (4096, 128, 1))
        a = padded((batch_size, seq_len, num_v_heads), (32, 1))
        b = padded((batch_size, seq_len, num_v_heads), (32, 1))
        state = torch.randn(
            3,
            num_v_heads,
            128,
            128,
            dtype=torch.float32,
            device="cuda",
        )
        state_indices = torch.tensor([2], dtype=torch.int32, device="cuda")
        intermediate = torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            128,
            128,
            dtype=torch.float32,
            device="cuda",
        )
        output = torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        entry = mock.Mock()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_decode_variant",
                wraps=cake_api.select_cake_gdn_decode_variant,
            ) as selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                return_value=entry,
            ),
        ):
            kernel = self._cake_decode_test_kernel(cake_api, output)
            actual = kernel._try_cake_decode(
                q=q,
                k=k,
                v=v,
                state=state,
                state_indices=state_indices,
                A_log=torch.zeros(num_v_heads, dtype=torch.float32, device="cuda"),
                a=a,
                dt_bias=torch.zeros(
                    num_v_heads, dtype=torch.float32, device="cuda"
                ),
                b=b,
                layer_id=0,
                disable_state_update=True,
                intermediate_state=intermediate,
                cache_steps=2,
            )

        self.assertIsNotNone(actual)
        self.assertTrue(selector.call_args.kwargs["strided_inputs"])
        self.assertEqual(len(entry.call_args.args), 14)
        self.assertEqual(
            entry.call_args.args[-3:],
            (batch_size * num_v_heads * 16, 1, 1),
        )

    def test_cake_exact_decode_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            run_gdn_attention_case(
                self,
                self.CAKE_DECODE_CASE,
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
            )
            eager_load_count = load_kernel.call_count
            self.assertGreater(eager_load_count, 0)
            run_gdn_cuda_graph_decode_case(
                self,
                self.CAKE_DECODE_CASE,
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
                cuda_graph_capture_batch_size=4,
            )
        self.assertGreater(load_kernel.call_count, eager_load_count)

    def test_cake_exact_prefill_updates_indexed_state_in_place(self):
        cake_api = self._cake_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=128,
        )
        initial_ssm_states = _ssm_states(fixture).clone()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            dispatcher = fixture.backend.linear_attn_backend.kernel_dispatcher
            with mock.patch.object(
                dispatcher,
                "extend",
                wraps=dispatcher.extend,
            ) as extend:
                actual = run_gdn_fixture_eager(fixture)
        expected = _pure_torch_gdn_reference(fixture, initial_ssm_states)
        cache_indices = _cache_indices(fixture)

        self.assertGreater(load_kernel.call_count, 0)
        extend.assert_called_once()
        self.assertIs(
            extend.call_args.kwargs["seq_lens_cpu"],
            fixture.forward_batch.extend_seq_lens_cpu,
        )
        self.assertEqual(extend.call_args.kwargs["layer_id"], 0)
        torch.testing.assert_close(actual, expected.output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

        # Prepare all stream-local Cake buffers before capture, then prove that
        # the same admitted non-CP route captures and replays on a caller stream.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        _ssm_states(fixture).copy_(initial_ssm_states)
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        _ssm_states(fixture).copy_(initial_ssm_states)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        _ssm_states(fixture).copy_(initial_ssm_states)
        torch.cuda.current_stream().synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, expected.output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

    def test_public_auto_cp_prefill_is_not_intercepted(self):
        cake_api = self._cake_api_or_skip()
        cake_cp_api = self._cake_cp_api_or_skip()
        from flashinfer.gdn_kernels.blackwell import cake_gdn_cp_prefill

        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CP_PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=256,
        )
        initial_ssm_states = _ssm_states(fixture).clone()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_prefill_variant",
                wraps=cake_api.select_cake_gdn_prefill_variant,
            ) as noncp_selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                wraps=cake_api.load_cake_gdn_kernel,
            ) as noncp_loader,
            mock.patch.object(
                cake_gdn_cp_prefill,
                "load_cake_gdn_cp_kernel",
                wraps=cake_cp_api.load_cake_gdn_cp_kernel,
            ) as cp_loader,
        ):
            actual = run_gdn_fixture_eager(fixture)
        expected = _pure_torch_gdn_reference(fixture, initial_ssm_states)
        cache_indices = _cache_indices(fixture)

        noncp_selector.assert_not_called()
        noncp_loader.assert_not_called()
        self.assertGreater(cp_loader.call_count, 0)
        torch.testing.assert_close(actual, expected.output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

    def test_traced_b1_t39_cp_prefill_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        cake_cp_api = self._cake_cp_api_or_skip()
        from flashinfer.gdn_kernels.blackwell import cake_gdn_cp_prefill

        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CP_PREFILL_T39_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=64,
            runner_batch_size=3,
        )
        _install_t39_padded_state_pool(fixture)
        initial_ssm_states = _ssm_states(fixture).clone()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_prefill_variant",
                wraps=cake_api.select_cake_gdn_prefill_variant,
            ) as noncp_selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                wraps=cake_api.load_cake_gdn_kernel,
            ) as noncp_loader,
            mock.patch.object(
                cake_gdn_cp_prefill,
                "load_cake_gdn_cp_kernel",
                wraps=cake_cp_api.load_cake_gdn_cp_kernel,
            ) as cp_loader,
        ):
            actual = run_gdn_fixture_eager(fixture)
            expected = _pure_torch_gdn_reference(fixture, initial_ssm_states)
            cache_indices = _cache_indices(fixture)

            noncp_selector.assert_not_called()
            noncp_loader.assert_not_called()
            self.assertGreater(cp_loader.call_count, 0)
            torch.testing.assert_close(actual, expected.output, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(
                _ssm_states(fixture)[cache_indices],
                expected.final_states[cache_indices],
                atol=1e-2,
                rtol=1e-2,
            )

            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            _ssm_states(fixture).copy_(initial_ssm_states)
            with (
                torch.no_grad(),
                torch.cuda.stream(capture_stream),
                forward_context(ForwardContext(attn_backend=fixture.backend)),
            ):
                fixture.backend.init_forward_metadata(fixture.forward_batch)
                fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
            capture_stream.synchronize()

            _ssm_states(fixture).copy_(initial_ssm_states)
            capture_stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with (
                torch.no_grad(),
                torch.cuda.stream(capture_stream),
                forward_context(ForwardContext(attn_backend=fixture.backend)),
            ):
                with torch.cuda.graph(graph, stream=capture_stream):
                    graph_output = fixture.actual_module(
                        fixture.forward_batch,
                        fixture.mixed_qkv,
                        fixture.a,
                        fixture.b,
                    )
            capture_stream.synchronize()

            _ssm_states(fixture).copy_(initial_ssm_states)
            torch.cuda.current_stream().synchronize()
            graph.replay()
            torch.cuda.synchronize()

        noncp_selector.assert_not_called()
        noncp_loader.assert_not_called()
        torch.testing.assert_close(
            graph_output, expected.output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )
        prepared = cake_gdn_cp_prefill._public_prepared
        self.assertIsNotNone(prepared)
        self.assertTrue(prepared._uses_fused_bf16_indexed_t39)
        self.assertIsNone(prepared._gather)
        self.assertIsNone(prepared._scatter)
        self.assertIsNone(prepared._checkpoint)
        self.assertIsNotNone(prepared._fused_state_carrier)
        self.assertEqual(prepared.plan.seq_lens, (39,))
        self.assertIn(prepared.plan.arch, ("sm_100a", "sm_103a"))

    def test_traced_b1_t39_cp_prefill_fused_bindings_are_capture_stable(self):
        self._cake_cp_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CP_PREFILL_T39_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=64,
            runner_batch_size=3,
        )
        _install_t39_padded_state_pool(fixture)
        dispatcher = fixture.backend.linear_attn_backend.kernel_dispatcher
        kernel = dispatcher.extend_kernel
        original_prefill = kernel._prefill_fn
        observed = []

        def record_prefill(**kwargs):
            capturing = torch.cuda.is_current_stream_capturing()
            initial_state = kwargs["initial_state"]
            output_state = kwargs["output_state"]
            checkpoints = kwargs["state_checkpoints"]
            observed.append(
                {
                    "capturing": capturing,
                    "initial_state": initial_state,
                    "initial_snapshot": (
                        None if capturing else initial_state.detach().clone()
                    ),
                    "output_state": output_state,
                    "output": kwargs["output"],
                    "state_indices": kwargs["state_indices"],
                    "checkpoints": checkpoints,
                }
            )
            return original_prefill(**kwargs)

        cache_indices = _cache_indices(fixture)
        refreshed_state = torch.randn_like(_ssm_states(fixture)[cache_indices]) * 0.01
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with (
            mock.patch.object(kernel, "_prefill_fn", side_effect=record_prefill),
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
            _ssm_states(fixture)[cache_indices] = refreshed_state
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        with (
            mock.patch.object(kernel, "_prefill_fn", side_effect=record_prefill),
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        self.assertEqual(len(observed), 3)
        warm_1, warm_2, captured = observed
        self.assertFalse(warm_1["capturing"])
        self.assertFalse(warm_2["capturing"])
        self.assertTrue(captured["capturing"])
        self.assertIs(warm_2["initial_state"], warm_1["initial_state"])
        self.assertIs(captured["initial_state"], warm_1["initial_state"])
        self.assertIs(warm_2["output_state"], warm_1["output_state"])
        self.assertIs(captured["output_state"], warm_1["output_state"])
        self.assertIs(warm_1["output_state"], warm_1["initial_state"])
        self.assertIs(warm_2["output"], warm_1["output"])
        self.assertIs(captured["output"], warm_1["output"])
        self.assertIs(warm_2["state_indices"], warm_1["state_indices"])
        self.assertIs(captured["state_indices"], warm_1["state_indices"])
        self.assertEqual(warm_1["state_indices"].dtype, torch.int32)
        self.assertEqual(tuple(warm_1["state_indices"].shape), (1,))
        self.assertIs(warm_2["checkpoints"], warm_1["checkpoints"])
        self.assertIs(captured["checkpoints"], warm_1["checkpoints"])
        self.assertIsNone(warm_1["checkpoints"])
        torch.testing.assert_close(
            warm_2["initial_snapshot"][cache_indices],
            refreshed_state,
            atol=0,
            rtol=0,
        )

    def test_traced_b1_t39_cp_prefill_fused_indices_require_warmup(self):
        self._cake_cp_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CP_PREFILL_T39_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=64,
            runner_batch_size=3,
        )
        _install_t39_padded_state_pool(fixture)
        kernel = fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        kernel._flashinfer_gdn_t39_state_indices = {}
        with (
            self.assertRaisesRegex(RuntimeError, "state indices must be warmed"),
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )

    def test_b1_t103_flashinfer_checkpoint_buffer_is_capture_stable(self):
        self._cake_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CHECKPOINT_PREFILL_B1_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=128,
            runner_batch_size=4,
        )
        batch = fixture.forward_batch
        batch.mamba_track_mask = torch.tensor(
            [True], dtype=torch.bool, device="cuda"
        )
        batch.mamba_track_indices = torch.tensor(
            [3], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            [107], dtype=torch.int64, device="cuda"
        )

        kernel = fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel
        original_prefill = kernel._prefill_fn
        checkpoints = []

        def record_prefill(**kwargs):
            checkpoints.append(kwargs["state_checkpoints"])
            return original_prefill(**kwargs)

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with (
            mock.patch.object(kernel, "_try_cake_prefill", return_value=None),
            mock.patch.object(kernel, "_prefill_fn", side_effect=record_prefill),
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        with (
            mock.patch.object(kernel, "_try_cake_prefill", return_value=None),
            mock.patch.object(kernel, "_prefill_fn", side_effect=record_prefill),
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        self.assertEqual(len(checkpoints), 3)
        warm_1, warm_2, captured = checkpoints
        self.assertIsNotNone(warm_1)
        self.assertEqual(tuple(warm_1.shape), (1, 8, 128, 128))
        self.assertIs(warm_2, warm_1)
        self.assertIs(captured, warm_1)

    def test_cake_exact_checkpoint_prefill_tracks_indexed_state(self):
        cake_api = self._cake_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CHECKPOINT_PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=192,
            runner_batch_size=14,
        )
        batch = fixture.forward_batch
        batch.mamba_track_mask = torch.tensor(
            [False, True, False, True, True, False, False],
            dtype=torch.bool,
            device="cuda",
        )
        batch.mamba_track_indices = torch.tensor(
            [8, 9, 10, 11, 12, 13, 14], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            [56, 100, 25, 120, 88, 80, 43],
            dtype=torch.int64,
            device="cuda",
        )

        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv = cache.conv[0].clone()
        initial_ssm = cache.temporal.clone()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            cake_output = run_gdn_fixture_eager(fixture)
        cake_tracked = cache.temporal[batch.mamba_track_indices].clone()
        cake_final = cache.temporal[_cache_indices(fixture)].clone()

        self.assertGreater(load_kernel.call_count, 0)

        # Prepare every stream-local output/workspace/checkpoint buffer, then
        # capture the same checkpoint route without a capture-time allocation.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        torch.cuda.current_stream().synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, cake_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            cache.temporal[batch.mamba_track_indices],
            cake_tracked,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            cache.temporal[_cache_indices(fixture)],
            cake_final,
            atol=1e-2,
            rtol=1e-2,
        )

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel = (
            TritonGDNKernel()
        )
        triton_output = run_gdn_fixture_eager(fixture)
        triton_tracked = cache.temporal[batch.mamba_track_indices]
        triton_final = cache.temporal[_cache_indices(fixture)]

        torch.testing.assert_close(cake_output, triton_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(cake_tracked, triton_tracked, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(cake_final, triton_final, atol=1e-2, rtol=1e-2)

    def test_cake_exact_b1_checkpoint_prefill_callthrough(self):
        cake_api = self._cake_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CHECKPOINT_PREFILL_B1_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=128,
            runner_batch_size=4,
        )
        batch = fixture.forward_batch
        batch.mamba_track_mask = torch.tensor([True], dtype=torch.bool, device="cuda")
        batch.mamba_track_indices = torch.tensor(
            [3], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            [107], dtype=torch.int64, device="cuda"
        )

        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv = cache.conv[0].clone()
        initial_ssm = cache.temporal.clone()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            cake_output = run_gdn_fixture_eager(fixture)
        cake_tracked = cache.temporal[batch.mamba_track_indices].clone()
        cake_final = cache.temporal[_cache_indices(fixture)].clone()

        self.assertGreater(load_kernel.call_count, 0)

        # The exact B1/T103 checkpoint route must reuse its caller-stream
        # buffers during capture and replay, just like the B7/T421 route.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        torch.cuda.current_stream().synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, cake_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            cache.temporal[batch.mamba_track_indices],
            cake_tracked,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            cache.temporal[_cache_indices(fixture)],
            cake_final,
            atol=1e-2,
            rtol=1e-2,
        )

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel = (
            TritonGDNKernel()
        )
        triton_output = run_gdn_fixture_eager(fixture)
        triton_tracked = cache.temporal[batch.mamba_track_indices]
        triton_final = cache.temporal[_cache_indices(fixture)]

        torch.testing.assert_close(cake_output, triton_output, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(cake_tracked, triton_tracked, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(cake_final, triton_final, atol=1e-2, rtol=1e-2)

    def test_cake_fp32_t1_decode_calls_public_backend(self):
        cake_api = self._cake_api_or_skip()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_decode_variant",
                wraps=cake_api.select_cake_gdn_decode_variant,
            ) as selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                wraps=cake_api.load_cake_gdn_kernel,
            ) as load_kernel,
        ):
            run_gdn_attention_case(
                self,
                self.CAKE_FP32_DECODE_CASE,
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
            )

        self.assertTrue(
            any(
                call.kwargs.get("batch_size") == 1
                and call.kwargs.get("seq_len") == 1
                and call.kwargs.get("num_q_heads") == 16
                and call.kwargs.get("num_v_heads") == 32
                and call.kwargs.get("state_dtype") == "float32"
                for call in selector.call_args_list
            )
        )
        self.assertGreater(load_kernel.call_count, 0)

    def test_cake_exact_verify_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            run_gdn_eagle_verify_case(
                self,
                self.CAKE_VERIFY_CASE,
                topk=1,
                spec_kind="frozen_kv_mtp",
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
            )
            eager_load_count = load_kernel.call_count
            self.assertGreater(eager_load_count, 0)
            run_gdn_eagle_verify_cuda_graph_case(
                self,
                self.CAKE_VERIFY_CASE,
                topk=1,
                spec_kind="frozen_kv_mtp",
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
                cuda_graph_capture_batch_size=8,
            )
        self.assertGreater(load_kernel.call_count, eager_load_count)

    def test_cake_traced_verify_batches_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        with mock.patch.object(
            cake_api,
            "select_cake_gdn_decode_variant",
            wraps=cake_api.select_cake_gdn_decode_variant,
        ) as selector, mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            for case in self.CAKE_TRACED_VERIFY_CASES:
                batch_size = len(case.prefix_lens)
                with self.subTest(batch_size=batch_size):
                    prior_selects = selector.call_count
                    prior_loads = load_kernel.call_count
                    run_gdn_eagle_verify_case(
                        self,
                        case,
                        topk=1,
                        spec_kind="frozen_kv_mtp",
                        head_k_dim=self.HEAD_DIM,
                        head_v_dim=self.HEAD_DIM,
                    )
                    self.assertGreater(selector.call_count, prior_selects)
                    self.assertGreater(load_kernel.call_count, prior_loads)
                    self.assertTrue(
                        any(
                            call.kwargs.get("batch_size") == batch_size
                            and call.kwargs.get("seq_len") == 4
                            for call in selector.call_args_list[prior_selects:]
                        )
                    )

                    prior_selects = selector.call_count
                    prior_loads = load_kernel.call_count
                    run_gdn_eagle_verify_cuda_graph_case(
                        self,
                        case,
                        topk=1,
                        spec_kind="frozen_kv_mtp",
                        head_k_dim=self.HEAD_DIM,
                        head_v_dim=self.HEAD_DIM,
                        cuda_graph_capture_batch_size=batch_size,
                    )
                    self.assertGreater(selector.call_count, prior_selects)
                    self.assertGreater(load_kernel.call_count, prior_loads)

    def test_prefill_tracked_state_checkpoints(self):
        fixture = build_gdn_attention_fixture(
            self,
            self.CHECKPOINT_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=320,
            runner_batch_size=6,
        )
        batch = fixture.forward_batch
        # Simulate the tracking metadata produced by the extra-buffer scheduler.
        # This test covers checkpoint mapping and state copies, not scheduler setup.
        batch.mamba_track_mask = torch.ones(3, dtype=torch.bool, device="cuda")
        batch.mamba_track_indices = torch.tensor(
            [4, 5, 6], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            # The final entry selects the second checkpoint at absolute S256.
            [64, 129, 257],
            dtype=torch.int64,
            device="cuda",
        )

        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv = cache.conv[0].clone()
        initial_ssm = cache.temporal.clone()
        flashinfer_output = run_gdn_fixture_eager(fixture)
        flashinfer_tracked = cache.temporal[batch.mamba_track_indices].clone()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel = (
            TritonGDNKernel()
        )
        triton_output = run_gdn_fixture_eager(fixture)
        triton_tracked = cache.temporal[batch.mamba_track_indices]

        torch.testing.assert_close(
            flashinfer_output, triton_output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            flashinfer_tracked, triton_tracked, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
