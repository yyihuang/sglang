# SPDX-License-Identifier: Apache-2.0

import argparse

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
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
