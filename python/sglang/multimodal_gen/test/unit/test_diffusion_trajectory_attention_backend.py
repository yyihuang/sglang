# SPDX-License-Identifier: Apache-2.0

import argparse
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    _extract_generation_time_s,
    build_server_kwargs,
)


def _args(**overrides):
    defaults = {
        "model_path": "model",
        "model_id": None,
        "backend": "sglang",
        "num_gpus": 1,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "pin_cpu_memory": False,
        "enable_cfg_parallel": False,
        "ulysses_degree": 1,
        "sp_degree": None,
        "reference_transformer_path": None,
        "candidate_transformer_path": None,
        "reference_component_path": [],
        "candidate_component_path": [],
        "reference_attention_backend": None,
        "candidate_attention_backend": None,
        "reference_component_attention_backend": [],
        "candidate_component_attention_backend": [],
    }
    return argparse.Namespace(**(defaults | overrides))


def test_build_server_kwargs_forwards_attention_backend_per_variant():
    args = _args(
        reference_attention_backend="dynamic_cudnn_sdpa",
        candidate_attention_backend="cake_nvfp4",
    )

    reference = build_server_kwargs(args, variant="reference")
    candidate = build_server_kwargs(args, variant="candidate")

    assert reference["attention_backend"] == "dynamic_cudnn_sdpa"
    assert candidate["attention_backend"] == "cake_nvfp4"


def test_build_server_kwargs_omits_unspecified_attention_backend():
    args = _args()

    assert "attention_backend" not in build_server_kwargs(args, variant="reference")
    assert "attention_backend" not in build_server_kwargs(args, variant="candidate")


def test_build_server_kwargs_forwards_component_attention_backends():
    args = _args(
        candidate_attention_backend="fa",
        candidate_component_attention_backend=[
            "transformer=cake_nvfp4",
            "transformer_2=fa",
        ],
    )

    candidate = build_server_kwargs(args, variant="candidate")

    assert candidate["attention_backend"] == "fa"
    assert candidate["component_attention_backends"] == {
        "transformer": "cake_nvfp4",
        "transformer_2": "fa",
    }


def test_extract_generation_time_prefers_populated_public_field():
    result = SimpleNamespace(
        generation_time=1.25,
        metrics={"total_duration_ms": 2500.0},
    )

    assert _extract_generation_time_s(result) == 1.25


def test_extract_generation_time_falls_back_to_scheduler_metrics():
    result = SimpleNamespace(
        generation_time=0.0,
        metrics={"total_duration_ms": 2500.0},
    )

    assert _extract_generation_time_s(result) == 2.5


def test_extract_generation_time_rejects_missing_measurement():
    result = SimpleNamespace(generation_time=0.0, metrics={})

    with pytest.raises(ValueError, match="neither a positive generation_time"):
        _extract_generation_time_s(result)
