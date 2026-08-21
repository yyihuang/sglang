# Copyright 2026 SGLang Team
import math

import pytest

from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _validate_wan_pv_nvfp4_exact_options,
)


_EXACT_SOFTMAX_SCALE = 1.0 / math.sqrt(128)


@pytest.mark.parametrize(
    ("softmax_scale", "max_seqlen_q", "max_seqlen_k"),
    [
        (None, None, None),
        (_EXACT_SOFTMAX_SCALE, 4800, 4800),
        (None, 4800, None),
        (_EXACT_SOFTMAX_SCALE, None, 4800),
    ],
)
def test_wan_pv_nvfp4_exact_options_accept_contract_values(
    softmax_scale, max_seqlen_q, max_seqlen_k
):
    _validate_wan_pv_nvfp4_exact_options(
        softmax_scale=softmax_scale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


@pytest.mark.parametrize(
    "softmax_scale",
    [
        0.1,
        math.nan,
        math.nextafter(_EXACT_SOFTMAX_SCALE, math.inf),
    ],
)
def test_wan_pv_nvfp4_exact_options_reject_other_softmax_scales(softmax_scale):
    with pytest.raises(ValueError, match="softmax_scale must be None or"):
        _validate_wan_pv_nvfp4_exact_options(
            softmax_scale=softmax_scale,
            max_seqlen_q=4800,
            max_seqlen_k=4800,
        )


@pytest.mark.parametrize(
    ("name", "max_seqlen_q", "max_seqlen_k"),
    [
        ("max_seqlen_q", 4799, 4800),
        ("max_seqlen_q", 4801, 4800),
        ("max_seqlen_k", 4800, 4799),
        ("max_seqlen_k", 4800, 4801),
    ],
)
def test_wan_pv_nvfp4_exact_options_reject_other_sequence_limits(
    name, max_seqlen_q, max_seqlen_k
):
    with pytest.raises(ValueError, match=rf"{name} must be None or 4800"):
        _validate_wan_pv_nvfp4_exact_options(
            softmax_scale=_EXACT_SOFTMAX_SCALE,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
