# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
from sglang.multimodal_gen.tools.aggregate_diffusion_attention_performance import (
    aggregate_paired_performance_reports,
)
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
    _cosine_similarity,
    _extract_wan_hybrid_hit_count,
    _extract_generation_time_s,
    _with_candidate_backend_hit_qualification,
    build_sampling_kwargs,
    build_server_kwargs,
    evaluate_candidate_backend_hit_qualification,
    evaluate_correctness_qualification,
    evaluate_performance_qualification,
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
        candidate_attention_backend="wan_hybrid",
    )

    reference = build_server_kwargs(args, variant="reference")
    candidate = build_server_kwargs(args, variant="candidate")

    assert reference["attention_backend"] == "dynamic_cudnn_sdpa"
    assert candidate["attention_backend"] == "wan_hybrid"


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
            "transformer=wan_hybrid",
            "transformer_2=fa",
        ],
    )

    candidate = build_server_kwargs(args, variant="candidate")

    assert candidate["attention_backend"] == "fa"
    assert candidate["component_attention_backends"] == {
        "transformer": "wan_hybrid",
        "transformer_2": "fa",
    }


def test_build_server_kwargs_forwards_attention_backend_config():
    args = _args(
        candidate_attention_backend="fa",
        candidate_attention_backend_config=(
            '{"wan_hybrid_min_timestep": 975}'
        ),
    )

    candidate = build_server_kwargs(args, variant="candidate")

    assert candidate["attention_backend_config"] == (
        '{"wan_hybrid_min_timestep":975}'
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


def test_extract_wan_hybrid_hit_count_requires_integer_metric():
    assert (
        _extract_wan_hybrid_hit_count(
            SimpleNamespace(metrics={"wan_hybrid_hit_count": 7})
        )
        == 7
    )
    for metrics in (
        {},
        {"wan_hybrid_hit_count": None},
        {"wan_hybrid_hit_count": True},
    ):
        assert _extract_wan_hybrid_hit_count(SimpleNamespace(metrics=metrics)) is None


def test_request_metrics_transports_wan_hybrid_hit_count():
    metrics = RequestMetrics("request")

    assert metrics.to_dict()["wan_hybrid_hit_count"] == 0
    metrics.wan_hybrid_hit_count = 11
    assert metrics.to_dict()["wan_hybrid_hit_count"] == 11


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


def _correctness_qualification(reference_results, candidate_results):
    return evaluate_correctness_qualification(
        summarize_cross_variant_metrics(reference_results, candidate_results),
        {
            "reference": summarize_run_repeatability(reference_results),
            "candidate": summarize_run_repeatability(candidate_results),
        },
    )


def test_correctness_qualification_passes_all_pairs_and_exact_repeatability():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(0.08), _generation_result(0.08)]

    qualification = _correctness_qualification(
        reference_results, candidate_results
    )

    assert qualification["passed"] is True
    assert qualification["failures"] == []
    assert qualification["thresholds"] == MODEL_QUALIFICATION_THRESHOLDS


def test_correctness_qualification_accepts_qualifying_output_frames():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(0.08), _generation_result(0.08)]
    cross_variant = summarize_cross_variant_metrics(
        reference_results, candidate_results
    )

    for comparison in cross_variant["comparisons"]:
        frame_metrics = comparison["output_metrics"]["all_frames_metrics"]
        assert frame_metrics["finite"] is True
        assert frame_metrics["within_tolerance"] is True
        assert frame_metrics["cosine_similarity"] >= MODEL_QUALIFICATION_THRESHOLDS[
            "cosine_min"
        ]
        assert frame_metrics["mae"] <= MODEL_QUALIFICATION_THRESHOLDS["mae_max"]

    qualification = evaluate_correctness_qualification(
        cross_variant,
        {
            "reference": summarize_run_repeatability(reference_results),
            "candidate": summarize_run_repeatability(candidate_results),
        },
    )

    assert qualification["passed"] is True
    assert qualification["failures"] == []


def test_correctness_qualification_rejects_finite_but_different_output_frames():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(0.08), _generation_result(0.08)]
    different_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    different_frame[:, :, 0] = 255
    for result in candidate_results:
        result.frames = [different_frame.copy()]
    cross_variant = summarize_cross_variant_metrics(
        reference_results, candidate_results
    )

    assert all(
        comparison["output_metrics"]["all_frames_metrics"]["finite"]
        for comparison in cross_variant["comparisons"]
    )
    qualification = evaluate_correctness_qualification(
        cross_variant,
        {
            "reference": summarize_run_repeatability(reference_results),
            "candidate": summarize_run_repeatability(candidate_results),
        },
    )

    assert qualification["passed"] is False
    assert {
        failure["reason"]
        for failure in qualification["failures"]
        if failure["scope"] == "cross_variant"
    } == {
        "output_frames_outside_atol_rtol",
        "output_frames_cosine_below_minimum",
        "output_frames_mae_above_maximum",
    }


def test_correctness_qualification_fails_closed_on_quality_and_repeatability():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(0.25), _generation_result(0.5)]

    qualification = _correctness_qualification(
        reference_results, candidate_results
    )

    assert qualification["passed"] is False
    reasons = {failure["reason"] for failure in qualification["failures"]}
    assert "mae_above_maximum" in reasons
    assert "repeatability_mismatch" in reasons


def test_correctness_qualification_fails_closed_on_non_finite_trajectory():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(), _generation_result()]
    candidate_results[0].trajectory_latents[0, 0, 0, 0] = float("nan")

    qualification = _correctness_qualification(
        reference_results, candidate_results
    )

    assert qualification["passed"] is False
    assert any(
        failure["reason"] == "non_finite_trajectory"
        for failure in qualification["failures"]
    )


def test_correctness_qualification_requires_same_instance_repeatability():
    result = _generation_result()
    qualification = evaluate_correctness_qualification(
        summarize_cross_variant_metrics([result], [result]),
        {
            "reference": summarize_run_repeatability([result]),
            "candidate": summarize_run_repeatability([result]),
        },
    )

    assert qualification["passed"] is False
    assert {
        failure["scope"] for failure in qualification["failures"]
    } == {"reference_repeatability", "candidate_repeatability"}


@pytest.mark.parametrize(
    ("speedup", "passed"),
    [(1.0, True), (1.01, True), (0.999, False), (float("nan"), False), (None, False)],
)
def test_performance_qualification_enforces_finite_speedup_floor(speedup, passed):
    qualification = evaluate_performance_qualification(
        {"wall_median_speedup": speedup}
    )

    assert qualification["passed"] is passed


@pytest.mark.parametrize(
    ("hit_counts", "passed"),
    [
        ([1], True),
        ([1, 4, 2], True),
        ([0], False),
        ([1, 0], False),
        ([None], False),
        (None, False),
    ],
)
def test_candidate_backend_hit_qualification_requires_every_run(hit_counts, passed):
    qualification = evaluate_candidate_backend_hit_qualification(hit_counts)

    assert qualification["passed"] is passed


def test_candidate_backend_hit_failure_is_part_of_overall_qualification():
    qualification = _with_candidate_backend_hit_qualification(
        {"passed": True, "failures": [], "thresholds": {}}, [0]
    )

    assert qualification["passed"] is False
    assert qualification["failures"] == [
        {
            "scope": "candidate_backend_hits",
            "reason": "candidate_hit_count_not_positive",
            "run_index": 0,
            "hit_count": 0,
        }
    ]


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
    assert kwargs["save_output"] is False


def _paired_performance_report(
    run_order: str,
    *,
    speedup: float = 1.0,
    hit_counts: list[int] | None = None,
):
    return {
        "schema_version": 2,
        "model_path": "/models/wan",
        "prompt": "fixed prompt",
        "seed": 42,
        "warmup_runs": 2,
        "measure_runs": 5,
        "comparison_mode": "performance",
        "run_order": run_order,
        "trajectory_capture_enabled": False,
        "server_kwargs": {
            "reference": {"attention_backend": "fa"},
            "candidate": {"attention_backend": "wan_hybrid"},
        },
        "backend_overrides": {
            "reference_attention_backend": "fa",
            "candidate_attention_backend": "wan_hybrid",
        },
        "sampling_kwargs": {
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
            "guidance_scale": 5.0,
            "guidance_scale_2": None,
            "return_frames": True,
            "return_trajectory_latents": False,
            "return_trajectory_decoded": False,
            "save_output": False,
        },
        "source_identity": {
            "sglang": {
                "git_revision": "a" * 40,
                "git_tree": "e" * 40,
                "git_clean": True,
                "git_status_sha256": "0" * 64,
                "module_file_sha256": "c" * 64,
            },
            "flashinfer": {
                "git_revision": "b" * 40,
                "git_tree": "f" * 40,
                "git_clean": True,
                "git_status_sha256": "0" * 64,
                "module_file_sha256": "d" * 64,
            },
        },
        "device_identity": {
            "index": 0,
            "name": "NVIDIA B200",
            "compute_capability": [10, 0],
            "total_memory_bytes": 192_000_000_000,
            "uuid": "GPU-fixed",
        },
        "runtime_provenance": {
            "python": "3.12.3",
            "torch": "2.9.0",
            "cuda": "13.0",
        },
        "candidate_generation": {
            "per_run_wan_hybrid_hit_count": hit_counts or [80, 80, 80, 80, 80]
        },
        "performance": {"wall_median_speedup": speedup},
    }


def test_paired_performance_qualification_passes_both_orders_at_floor():
    result = aggregate_paired_performance_reports(
        _paired_performance_report("reference-first", speedup=1.0),
        _paired_performance_report("candidate-first", speedup=1.01),
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["paired_speedups"] == {
        "reference_first": 1.0,
        "candidate_first": 1.01,
    }


def test_paired_performance_qualification_requires_opposite_orders():
    result = aggregate_paired_performance_reports(
        _paired_performance_report("candidate-first"),
        _paired_performance_report("candidate-first"),
    )

    assert result["passed"] is False
    assert {
        (failure["report"], failure["field"], failure["reason"])
        for failure in result["failures"]
    } >= {("reference_first", "run_order", "unexpected_value")}


@pytest.mark.parametrize(
    ("mutation", "field", "reason"),
    [
        (
            lambda report: report.update(comparison_mode="correctness"),
            "comparison_mode",
            "performance_mode_required",
        ),
        (
            lambda report: report.update(trajectory_capture_enabled=True),
            "trajectory_capture_enabled",
            "trajectory_capture_must_be_explicitly_disabled",
        ),
        (
            lambda report: report["sampling_kwargs"].update(save_output=True),
            "sampling_kwargs.save_output",
            "output_saving_must_be_explicitly_disabled",
        ),
        (
            lambda report: report["source_identity"]["sglang"].update(
                git_clean=False,
                git_status_sha256="1" * 64,
            ),
            "source_identity.sglang.git_clean",
            "clean_source_tree_required",
        ),
        (
            lambda report: report["candidate_generation"].update(
                per_run_wan_hybrid_hit_count=[80, 0]
            ),
            "candidate_generation.per_run_wan_hybrid_hit_count[1]",
            "candidate_hit_count_not_positive",
        ),
        (
            lambda report: report["candidate_generation"].update(
                per_run_wan_hybrid_hit_count=[80]
            ),
            "candidate_generation.per_run_wan_hybrid_hit_count",
            "candidate_hit_count_length_mismatch",
        ),
        (
            lambda report: report["performance"].update(
                wall_median_speedup=0.999
            ),
            "performance.wall_median_speedup",
            "speedup_below_minimum",
        ),
    ],
)
def test_paired_performance_qualification_fails_closed_per_order(
    mutation, field, reason
):
    reference_first = _paired_performance_report("reference-first")
    mutation(reference_first)

    result = aggregate_paired_performance_reports(
        reference_first,
        _paired_performance_report("candidate-first"),
    )

    assert result["passed"] is False
    assert any(
        failure["report"] == "reference_first"
        and failure["field"] == field
        and failure["reason"] == reason
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    "field",
    ["server_kwargs", "source_identity", "device_identity", "runtime_provenance"],
)
def test_paired_performance_qualification_requires_matching_identity(field):
    reference_first = _paired_performance_report("reference-first")
    candidate_first = _paired_performance_report("candidate-first")
    candidate_first[field] = copy.deepcopy(candidate_first[field])
    candidate_first[field]["mismatch"] = True

    result = aggregate_paired_performance_reports(
        reference_first, candidate_first
    )

    assert result["passed"] is False
    mismatch = next(
        failure
        for failure in result["failures"]
        if failure["report"] == "paired" and failure["field"] == field
    )
    assert mismatch["reason"] == "paired_value_mismatch"
    assert mismatch["reference_first_sha256"] != mismatch[
        "candidate_first_sha256"
    ]


def test_paired_performance_qualification_requires_device_uuid():
    reference_first = _paired_performance_report("reference-first")
    reference_first["device_identity"]["uuid"] = None

    result = aggregate_paired_performance_reports(
        reference_first,
        _paired_performance_report("candidate-first"),
    )

    assert result["passed"] is False
    assert any(
        failure["report"] == "reference_first"
        and failure["field"] == "device_identity.uuid"
        and failure["reason"] == "physical_device_uuid_required"
        for failure in result["failures"]
    )
