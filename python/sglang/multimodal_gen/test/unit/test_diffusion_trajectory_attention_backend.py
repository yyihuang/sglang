# SPDX-License-Identifier: Apache-2.0

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    _cosine_similarity,
    _extract_generation_time_s,
    build_sampling_kwargs,
    build_server_kwargs,
    summarize_cross_variant_metrics,
    summarize_run_repeatability,
)


def _args(**overrides):
    defaults = {
        "model_path": "model",
        "model_id": None,
        "backend": "sglang",
        "num_gpus": 1,
        "master_port": None,
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
        "reference_attention_backend_config": None,
        "candidate_attention_backend_config": None,
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


def test_build_server_kwargs_forwards_master_port():
    args = _args(master_port=31005)

    assert build_server_kwargs(args, variant="reference")["master_port"] == 31005
    assert build_server_kwargs(args, variant="candidate")["master_port"] == 31005


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


def test_build_server_kwargs_forwards_attention_backend_config():
    args = _args(
        candidate_attention_backend="fa",
        candidate_attention_backend_config=(
            '{"cake_nvfp4_min_timestep": 975}'
        ),
    )

    candidate = build_server_kwargs(args, variant="candidate")

    assert candidate["attention_backend_config"] == (
        '{"cake_nvfp4_min_timestep":975}'
    )


def test_build_server_kwargs_rejects_non_object_attention_backend_config():
    args = _args(candidate_attention_backend_config="[975]")

    with pytest.raises(ValueError, match="must decode to a JSON object"):
        build_server_kwargs(args, variant="candidate")


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


def _generation_result(latent_offset=0.0, frame_offset=0):
    return SimpleNamespace(
        trajectory_latents=torch.tensor(
            [[[[1.0 + latent_offset, 2.0], [3.0, 4.0]]]]
        ),
        trajectory_timesteps=torch.tensor([1.0]),
        frames=[np.full((2, 2, 3), 10 + frame_offset, dtype=np.uint8)],
        samples=None,
        output_file_path=None,
    )


def test_summarize_run_repeatability_reports_same_variant_envelope():
    repeatability = summarize_run_repeatability(
        [_generation_result(), _generation_result(0.25, 2)]
    )

    assert repeatability["available"] is True
    assert repeatability["num_runs"] == 2
    assert repeatability["pairing"] == "all-pairs"
    assert repeatability["num_pairs"] == 1
    assert repeatability["envelope"]["max_selected_trajectory_mae"] == pytest.approx(
        0.0625
    )
    assert repeatability["envelope"]["max_all_frames_mae"] == pytest.approx(2.0)


def test_summarize_run_repeatability_uses_every_pair_and_every_step():
    run0 = _generation_result()
    run1 = _generation_result(1.0, 2)
    run2 = _generation_result(-1.0, -2)
    run1.trajectory_latents = torch.cat(
        (run1.trajectory_latents, run0.trajectory_latents), dim=1
    )
    run2.trajectory_latents = torch.cat(
        (run2.trajectory_latents, run0.trajectory_latents), dim=1
    )
    run0.trajectory_latents = torch.cat(
        (run0.trajectory_latents, run0.trajectory_latents), dim=1
    )
    for result in (run0, run1, run2):
        result.trajectory_timesteps = torch.tensor([2.0, 1.0])

    repeatability = summarize_run_repeatability(
        [run0, run1, run2], step_index=-1
    )

    assert repeatability["num_pairs"] == 3
    assert repeatability["envelope"]["max_selected_trajectory_mae"] == 0.0
    assert repeatability["envelope"][
        "max_all_steps_trajectory_mae"
    ] == pytest.approx(0.5)
    assert repeatability["envelope"]["max_all_frames_mae"] == pytest.approx(4.0)


def test_summarize_cross_variant_metrics_uses_cross_product():
    summary = summarize_cross_variant_metrics(
        [_generation_result(), _generation_result(0.25, 1)],
        [
            _generation_result(0.5, 2),
            _generation_result(0.75, 3),
            _generation_result(1.0, 4),
        ],
    )

    assert summary["pairing"] == "cross-product"
    assert summary["num_pairs"] == 6
    assert summary["envelope"]["max_all_frames_mae"] == pytest.approx(4.0)


def test_summarize_run_repeatability_requires_two_runs():
    repeatability = summarize_run_repeatability([_generation_result()])

    assert repeatability == {
        "available": False,
        "num_runs": 1,
        "reason": "repeatability requires at least two measured runs",
    }


def test_cosine_similarity_is_clamped_to_valid_metric_range(monkeypatch):
    monkeypatch.setattr(
        torch.nn.functional,
        "cosine_similarity",
        lambda *args, **kwargs: torch.tensor(1.0001),
    )

    assert _cosine_similarity(torch.ones(2), torch.ones(2)) == 1.0


def _sampling_args(**overrides):
    defaults = {
        "prompt": "prompt",
        "width": 640,
        "height": 384,
        "num_inference_steps": 12,
        "guidance_scale": 4.0,
        "seed": 0,
        "return_trajectory_decoded": False,
        "num_frames": 17,
        "guidance_scale_2": 3.0,
        "comparison_mode": "correctness",
    }
    return argparse.Namespace(**(defaults | overrides))


def test_build_sampling_kwargs_captures_trajectory_for_correctness():
    kwargs = build_sampling_kwargs(_sampling_args())

    assert kwargs["return_frames"] is True
    assert kwargs["return_trajectory_latents"] is True


def test_build_sampling_kwargs_disables_trajectory_for_performance():
    kwargs = build_sampling_kwargs(
        _sampling_args(
            comparison_mode="performance", return_trajectory_decoded=True
        )
    )

    assert kwargs["return_frames"] is True
    assert kwargs["return_trajectory_latents"] is False
    assert kwargs["return_trajectory_decoded"] is False
