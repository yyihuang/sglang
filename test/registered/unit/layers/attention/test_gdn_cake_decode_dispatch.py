import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

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


class TestCakeGDNDecodeDispatch(unittest.TestCase):
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
