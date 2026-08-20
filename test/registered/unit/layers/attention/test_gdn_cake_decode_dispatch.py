import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _UnsupportedError(NotImplementedError):
    pass


class _CakeAPI:
    CakeGDNUnsupportedError = _UnsupportedError

    def __init__(self):
        self.select_cake_gdn_decode_variant = MagicMock(
            return_value=SimpleNamespace(
                route_id="cake.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp",
                variant_name="decode_bf16_verify_t4_tile16",
            )
        )
        self.load_cake_gdn_kernel = MagicMock()


def _kernel_and_inputs(batch_size: int):
    api = _CakeAPI()
    entry = MagicMock()
    output = torch.empty(batch_size, 4, 8, 128, dtype=torch.bfloat16)
    kernel = object.__new__(FlashInferGDNKernel)
    kernel._cake_gdn_api = api
    kernel._cake_gdn_arch = "sm_100a"
    kernel._cake_gdn_entries = {"decode_bf16_verify_t4_tile16": entry}
    kernel._cake_gdn_logged_routes = set()
    kernel._cake_output_buffer = MagicMock(return_value=output)
    kernel._cake_fp32_dt_bias = MagicMock(side_effect=lambda value, **_: value)

    q = torch.empty(batch_size, 4, 4, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(batch_size, 4, 8, 128, dtype=torch.bfloat16)
    state = torch.empty(batch_size + 3, 8, 128, 128, dtype=torch.bfloat16)
    state_indices = torch.arange(batch_size, dtype=torch.int32).flip(0).contiguous()
    A_log = torch.empty(8, dtype=torch.float32)
    a = torch.empty(batch_size, 4, 8, dtype=torch.bfloat16)
    dt_bias = torch.empty(8, dtype=torch.float32)
    b = torch.empty_like(a)
    intermediate_state = torch.empty(
        batch_size, 4, 8, 128, 128, dtype=torch.bfloat16
    )
    inputs = dict(
        q=q,
        k=k,
        v=v,
        state=state,
        state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        layer_id=7,
        disable_state_update=True,
        intermediate_state=intermediate_state,
        cache_steps=4,
    )
    return kernel, api, entry, inputs, output


def _fp32_t1_kernel_and_inputs():
    api = _CakeAPI()
    api.select_cake_gdn_decode_variant.return_value = SimpleNamespace(
        route_id="cake.gdn_decode.indexed_fp32_t1_splitv8",
        variant_name="decode_fp32_t1_splitv8",
    )
    entry = MagicMock()
    output = torch.empty(1, 1, 32, 128, dtype=torch.bfloat16)
    kernel = object.__new__(FlashInferGDNKernel)
    kernel._cake_gdn_api = api
    kernel._cake_gdn_arch = "sm_100a"
    kernel._cake_gdn_entries = {"decode_fp32_t1_splitv8": entry}
    kernel._cake_gdn_logged_routes = set()
    kernel._cake_output_buffer = MagicMock(return_value=output)
    kernel._cake_fp32_dt_bias = MagicMock(side_effect=lambda value, **_: value)

    inputs = dict(
        q=torch.empty(1, 1, 16, 128, dtype=torch.bfloat16),
        k=torch.empty(1, 1, 16, 128, dtype=torch.bfloat16),
        v=torch.empty(1, 1, 32, 128, dtype=torch.bfloat16),
        state=torch.empty(4, 32, 128, 128, dtype=torch.float32),
        state_indices=torch.tensor([3], dtype=torch.int32),
        A_log=torch.empty(32, dtype=torch.float32),
        a=torch.empty(1, 1, 32, dtype=torch.bfloat16),
        dt_bias=torch.empty(32, dtype=torch.float32),
        b=torch.empty(1, 1, 32, dtype=torch.bfloat16),
        layer_id=7,
        disable_state_update=False,
        intermediate_state=None,
        cache_steps=0,
    )
    return kernel, api, entry, inputs, output


def _prefill_metadata_kernel():
    kernel = object.__new__(FlashInferGDNKernel)
    kernel._flashinfer_gdn_prefill_metadata = {}
    return kernel


class TestCakeGDNDecodeDispatch(unittest.TestCase):
    def test_prefill_metadata_refreshes_stable_buffers_after_source_change(self):
        kernel = _prefill_metadata_kernel()
        query_start_loc = torch.tensor([0, 39], dtype=torch.int32)
        cache_indices = torch.tensor([-1], dtype=torch.int32)
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            indices_1, cu_seqlens_1 = kernel._flashinfer_prefill_metadata(
                query_start_loc, cache_indices
            )
            indices_version_1 = int(indices_1._version)
            cu_version_1 = int(cu_seqlens_1._version)

            query_start_loc.copy_(torch.tensor([0, 31], dtype=torch.int32))
            cache_indices.copy_(torch.tensor([3], dtype=torch.int32))
            indices_2, cu_seqlens_2 = kernel._flashinfer_prefill_metadata(
                query_start_loc, cache_indices
            )

        self.assertIs(indices_2, indices_1)
        self.assertIs(cu_seqlens_2, cu_seqlens_1)
        self.assertGreater(int(indices_2._version), indices_version_1)
        self.assertGreater(int(cu_seqlens_2._version), cu_version_1)
        torch.testing.assert_close(indices_2, torch.tensor([3], dtype=torch.int64))
        torch.testing.assert_close(
            cu_seqlens_2, torch.tensor([0, 31], dtype=torch.int64)
        )

    def test_prefill_metadata_capture_reuses_warmed_objects_unchanged(self):
        kernel = _prefill_metadata_kernel()
        query_start_loc = torch.tensor([0, 39], dtype=torch.int32)
        cache_indices = torch.tensor([2], dtype=torch.int32)
        stream = SimpleNamespace(cuda_stream=23)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ) as capturing,
        ):
            warmed_indices, warmed_cu = kernel._flashinfer_prefill_metadata(
                query_start_loc, cache_indices
            )
            warmed_versions = (
                int(warmed_indices._version),
                int(warmed_cu._version),
            )
            capturing.return_value = True
            captured_indices, captured_cu = kernel._flashinfer_prefill_metadata(
                query_start_loc, cache_indices
            )

        self.assertIs(captured_indices, warmed_indices)
        self.assertIs(captured_cu, warmed_cu)
        self.assertEqual(
            (int(captured_indices._version), int(captured_cu._version)),
            warmed_versions,
        )

    def test_prefill_metadata_capture_without_warm_fails_closed(self):
        kernel = _prefill_metadata_kernel()
        query_start_loc = torch.tensor([0, 39], dtype=torch.int32)
        cache_indices = torch.tensor([2], dtype=torch.int32)
        stream = SimpleNamespace(cuda_stream=29)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            self.assertRaisesRegex(RuntimeError, "must be warmed"),
        ):
            kernel._flashinfer_prefill_metadata(query_start_loc, cache_indices)

    def test_fp32_t1_calls_public_selector_and_entry(self):
        kernel, api, entry, inputs, output = _fp32_t1_kernel_and_inputs()

        result = kernel._try_cake_decode(**inputs)

        self.assertEqual(tuple(result.shape), (1, 1, 32, 128))
        api.select_cake_gdn_decode_variant.assert_called_once_with(
            arch="sm_100a",
            batch_size=1,
            io_dtype="bfloat16",
            state_dtype="float32",
            head_size=128,
            layout="pretranspose",
            num_k_heads=16,
            num_q_heads=16,
            num_v_heads=32,
            scale=128**-0.5,
            seq_len=1,
            use_qk_l2norm=True,
            strided_inputs=True,
            disable_state_update=False,
            cache_intermediate_states=False,
            cache_steps=0,
        )
        entry.assert_called_once()
        args = entry.call_args.args
        self.assertIs(args[0], inputs["q"])
        self.assertIs(args[3], inputs["state"])
        self.assertIs(args[8], output)
        self.assertIs(args[9], inputs["state_indices"])
        self.assertIs(args[10], inputs["state_indices"])
        self.assertEqual(args[11:14], (128, 1, 1))

    def test_fp32_cached_mtp_fails_closed_before_public_selector(self):
        kernel, api, entry, inputs, _ = _fp32_t1_kernel_and_inputs()
        inputs.update(
            q=torch.empty(1, 4, 16, 128, dtype=torch.bfloat16),
            k=torch.empty(1, 4, 16, 128, dtype=torch.bfloat16),
            v=torch.empty(1, 4, 32, 128, dtype=torch.bfloat16),
            a=torch.empty(1, 4, 32, dtype=torch.bfloat16),
            b=torch.empty(1, 4, 32, dtype=torch.bfloat16),
            disable_state_update=True,
            intermediate_state=torch.empty(
                1, 4, 32, 128, 128, dtype=torch.float32
            ),
            cache_steps=4,
        )

        result = kernel._try_cake_decode(**inputs)

        self.assertIsNone(result)
        api.select_cake_gdn_decode_variant.assert_not_called()
        entry.assert_not_called()

    def test_traced_tp4_verify_batches_call_public_selector_and_frozen_grid(self):
        for batch_size in (1, 4, 6, 7):
            with self.subTest(batch_size=batch_size):
                kernel, api, entry, inputs, output = _kernel_and_inputs(batch_size)

                result = kernel._try_cake_decode(**inputs)

                self.assertEqual(tuple(result.shape), (1, batch_size * 4, 8, 128))
                api.select_cake_gdn_decode_variant.assert_called_once_with(
                    arch="sm_100a",
                    batch_size=batch_size,
                    io_dtype="bfloat16",
                    state_dtype="bfloat16",
                    head_size=128,
                    layout="pretranspose",
                    num_k_heads=4,
                    num_q_heads=4,
                    num_v_heads=8,
                    scale=128**-0.5,
                    seq_len=4,
                    use_qk_l2norm=True,
                    strided_inputs=True,
                    disable_state_update=True,
                    cache_intermediate_states=True,
                    cache_steps=4,
                )
                entry.assert_called_once()
                args = entry.call_args.args
                self.assertIs(args[0], inputs["q"])
                self.assertIs(args[3], inputs["state"])
                self.assertIs(args[8], output)
                self.assertIs(args[9], inputs["intermediate_state"])
                self.assertIs(args[10], inputs["state_indices"])
                self.assertIs(args[11], inputs["state_indices"])
                self.assertEqual(args[12:15], (batch_size * 64, 1, 1))


if __name__ == "__main__":
    unittest.main()
