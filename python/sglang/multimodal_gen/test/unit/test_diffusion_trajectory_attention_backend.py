# SPDX-License-Identifier: Apache-2.0

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
    _cosine_similarity,
    _extract_generation_time_s,
    _extract_wan_hybrid_coverage,
    _extract_wan_hybrid_hit_count,
    _request_sampling_kwargs,
    _with_candidate_backend_hit_qualification,
    build_sampling_kwargs,
    build_server_kwargs,
    evaluate_candidate_backend_hit_qualification,
    evaluate_correctness_qualification,
    evaluate_dual_order_performance_qualification,
    evaluate_performance_qualification,
    run_variant,
    summarize_cross_variant_metrics,
    summarize_result_output,
    summarize_run_repeatability,
    validate_qualification_protocol,
    validate_server_port_isolation,
)


def _args(**overrides):
    defaults = {
        "model_path": "model",
        "model_id": None,
        "backend": "sglang",
        "num_gpus": 1,
        "master_port": None,
        "reference_scheduler_port": None,
        "candidate_scheduler_port": None,
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


def test_build_server_kwargs_forwards_isolated_scheduler_ports():
    args = _args(
        master_port=31005,
        reference_scheduler_port=56000,
        candidate_scheduler_port=56001,
    )

    reference = build_server_kwargs(args, variant="reference")
    candidate = build_server_kwargs(args, variant="candidate")

    assert reference["scheduler_port"] == 56000
    assert candidate["scheduler_port"] == 56001
    assert reference["strict_ports"] is True
    assert candidate["strict_ports"] is True


def test_validate_server_port_isolation_requires_distinct_explicit_ports():
    validate_server_port_isolation(
        master_port=31005,
        reference_scheduler_port=56000,
        candidate_scheduler_port=56001,
        enforce_qualification=True,
    )

    with pytest.raises(ValueError, match="requires explicit isolated ports"):
        validate_server_port_isolation(
            master_port=31005,
            reference_scheduler_port=None,
            candidate_scheduler_port=56001,
            enforce_qualification=True,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        validate_server_port_isolation(
            master_port=31005,
            reference_scheduler_port=56000,
            candidate_scheduler_port=56000,
            enforce_qualification=True,
        )


def test_build_server_kwargs_omits_unspecified_attention_backend():
    args = _args()

    assert "attention_backend" not in build_server_kwargs(args, variant="reference")
    assert "attention_backend" not in build_server_kwargs(args, variant="candidate")


def test_request_sampling_kwargs_are_unique_and_do_not_mutate_fixed_input():
    fixed = {"prompt": "A curious raccoon", "seed": 4254}

    reference = _request_sampling_kwargs(
        fixed, variant_name="reference", phase="measure", run_index=0
    )
    candidate = _request_sampling_kwargs(
        fixed, variant_name="candidate", phase="measure", run_index=0
    )
    next_candidate = _request_sampling_kwargs(
        fixed, variant_name="candidate", phase="measure", run_index=1
    )

    assert "request_id" not in fixed
    assert reference["request_id"] != candidate["request_id"]
    assert candidate["request_id"] != next_candidate["request_id"]
    assert {
        key: value for key, value in reference.items() if key != "request_id"
    } == fixed


def test_run_variant_propagates_unique_request_ids_in_local_mode(monkeypatch):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    generated_request_ids = []
    from_pretrained_calls = []

    class _FakeGenerator:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def generate(self, *, sampling_params_kwargs):
            request_id = sampling_params_kwargs["request_id"]
            generated_request_ids.append(request_id)
            return SimpleNamespace(
                frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
                samples=None,
                output_file_path=None,
                peak_memory_mb=0.0,
                metrics={
                    "request_id": request_id,
                    "total_duration_ms": 1.0,
                    "wan_hybrid_hit_count": 0,
                    "wan_hybrid_coverage": {
                        "request_id": request_id,
                        "expected_hit_count": 0,
                    },
                },
            )

    def fake_from_pretrained(**kwargs):
        from_pretrained_calls.append(kwargs)
        return _FakeGenerator()

    monkeypatch.setattr(
        DiffGenerator, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    fixed = {"prompt": "A curious raccoon", "seed": 4254}

    result = run_variant(
        variant_name="candidate",
        server_kwargs={},
        sampling_kwargs=fixed,
        fp4_gemm_backend=None,
        warmup_runs=2,
        measure_runs=5,
    )

    assert from_pretrained_calls == [{"local_mode": True}]
    assert len(generated_request_ids) == 7
    assert len(set(generated_request_ids)) == 7
    assert result["per_run_request_id"] == generated_request_ids[2:]
    assert fixed == {"prompt": "A curious raccoon", "seed": 4254}


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
        candidate_attention_backend_config=('{"wan_hybrid_min_timestep": 975}'),
    )

    candidate = build_server_kwargs(args, variant="candidate")

    assert candidate["attention_backend_config"] == ('{"wan_hybrid_min_timestep":975}')


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
    metrics.wan_hybrid_coverage = {"expected_hit_count": 11}
    assert metrics.to_dict()["wan_hybrid_coverage"] == {"expected_hit_count": 11}


def test_extract_wan_hybrid_coverage_requires_object_metric():
    coverage = {"schema_version": 1, "expected_hit_count": 2}
    result = SimpleNamespace(metrics={"wan_hybrid_coverage": coverage})

    assert _extract_wan_hybrid_coverage(result) == coverage
    assert _extract_wan_hybrid_coverage(SimpleNamespace(metrics={})) is None


def test_output_summary_hashes_materialized_frame_bytes():
    result = SimpleNamespace(
        frames=[np.arange(12, dtype=np.uint8).reshape(2, 2, 3)],
        samples=None,
        output_file_path=None,
    )

    summary = summarize_result_output(result)

    assert len(summary["sha256"]) == 64
    assert summary["num_frames"] == 1
    assert summary["frame_shapes"] == [[2, 2, 3]]
    assert summary["frame_dtypes"] == ["uint8"]
    assert summary["finite"] is True


def _generation_result(latent_offset=0.0, frame_offset=0):
    return SimpleNamespace(
        trajectory_latents=torch.tensor([[[[1.0 + latent_offset, 2.0], [3.0, 4.0]]]]),
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

    repeatability = summarize_run_repeatability([run0, run1, run2], step_index=-1)

    assert repeatability["num_pairs"] == 3
    assert repeatability["envelope"]["max_selected_trajectory_mae"] == 0.0
    assert repeatability["envelope"]["max_all_steps_trajectory_mae"] == pytest.approx(
        0.5
    )
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

    qualification = _correctness_qualification(reference_results, candidate_results)

    assert qualification["passed"] is True
    assert qualification["failures"] == []
    assert qualification["thresholds"] == MODEL_QUALIFICATION_THRESHOLDS


def test_correctness_qualification_fails_closed_on_quality_and_repeatability():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(0.25), _generation_result(0.5)]

    qualification = _correctness_qualification(reference_results, candidate_results)

    assert qualification["passed"] is False
    reasons = {failure["reason"] for failure in qualification["failures"]}
    assert "mae_above_maximum" in reasons
    assert "repeatability_mismatch" in reasons


def test_correctness_qualification_fails_closed_on_non_finite_trajectory():
    reference_results = [_generation_result(), _generation_result()]
    candidate_results = [_generation_result(), _generation_result()]
    candidate_results[0].trajectory_latents[0, 0, 0, 0] = float("nan")

    qualification = _correctness_qualification(reference_results, candidate_results)

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
    assert {failure["scope"] for failure in qualification["failures"]} == {
        "reference_repeatability",
        "candidate_repeatability",
    }


@pytest.mark.parametrize(
    ("speedup", "passed"),
    [
        (1.0, True),
        (1.01, True),
        (0.999, False),
        (float("nan"), False),
        (None, False),
        (True, False),
    ],
)
def test_performance_qualification_enforces_finite_speedup_floor(speedup, passed):
    qualification = evaluate_performance_qualification({"wall_median_speedup": speedup})

    assert qualification["passed"] is passed


def _performance_order_result(
    *,
    speedup=1.0,
    hit_counts=None,
    warmup_runs=2,
    measure_runs=5,
):
    if hit_counts is None:
        hit_counts = [1] * measure_runs
    generation = {
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
    }
    return {
        "reference_generation": dict(generation),
        "candidate_generation": generation
        | {
            "per_run_wan_hybrid_hit_count": hit_counts,
            "per_run_wan_hybrid_expected_hit_count": [1] * measure_runs,
        },
        "performance": {"wall_median_speedup": speedup},
    }


def test_qualification_protocol_requires_two_five_and_dual_performance_order():
    validate_qualification_protocol(
        comparison_mode="performance",
        run_order="both",
        warmup_runs=2,
        measure_runs=5,
    )

    invalid_protocols = (
        {"run_order": "both", "warmup_runs": 1, "measure_runs": 5},
        {"run_order": "both", "warmup_runs": 2, "measure_runs": 4},
        {
            "run_order": "reference-first",
            "warmup_runs": 2,
            "measure_runs": 5,
        },
    )
    for protocol in invalid_protocols:
        with pytest.raises(ValueError, match="Invalid qualification protocol"):
            validate_qualification_protocol(comparison_mode="performance", **protocol)

    validate_qualification_protocol(
        comparison_mode="correctness",
        run_order="reference-first",
        warmup_runs=2,
        measure_runs=5,
    )
    with pytest.raises(ValueError, match="warmup_runs must be >= 2"):
        validate_qualification_protocol(
            comparison_mode="correctness",
            run_order="reference-first",
            warmup_runs=1,
            measure_runs=5,
        )


def test_dual_order_performance_qualification_requires_both_passing_orders():
    qualification = evaluate_dual_order_performance_qualification(
        {
            "reference-first": _performance_order_result(speedup=1.01),
            "candidate-first": _performance_order_result(speedup=1.02),
        }
    )

    assert qualification["passed"] is True
    assert qualification["failures"] == []


def test_dual_order_performance_qualification_fails_closed():
    qualification = evaluate_dual_order_performance_qualification(
        {
            "reference-first": _performance_order_result(
                speedup=0.99, hit_counts=[1, 1, 0, 1]
            ),
        }
    )

    assert qualification["passed"] is False
    reasons = {failure["reason"] for failure in qualification["failures"]}
    assert reasons == {
        "wall_median_speedup_below_minimum",
        "candidate_hit_count_cardinality_mismatch",
        "candidate_expected_hit_count_cardinality_mismatch",
        "candidate_hit_count_not_positive",
        "candidate_hit_count_mismatch",
        "missing_run_order_result",
    }


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
def test_candidate_backend_hit_qualification_requires_exact_expected_hits(
    hit_counts, passed
):
    expected_hit_counts = (
        list(hit_counts)
        if isinstance(hit_counts, list) and all(hit_count for hit_count in hit_counts)
        else [1] * len(hit_counts) if isinstance(hit_counts, list) else None
    )
    qualification = evaluate_candidate_backend_hit_qualification(
        hit_counts, expected_hit_counts
    )

    assert qualification["passed"] is passed


def test_candidate_backend_hit_failure_is_part_of_overall_qualification():
    qualification = _with_candidate_backend_hit_qualification(
        {"passed": True, "failures": [], "thresholds": {}}, [0], [1]
    )

    assert qualification["passed"] is False
    assert qualification["failures"] == [
        {
            "scope": "candidate_backend_hits",
            "reason": "candidate_hit_count_not_positive",
            "run_index": 0,
            "hit_count": 0,
        },
        {
            "scope": "candidate_backend_hits",
            "reason": "candidate_hit_count_mismatch",
            "run_index": 0,
            "expected_hit_count": 1,
            "actual_hit_count": 0,
        },
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
        _sampling_args(comparison_mode="performance", return_trajectory_decoded=True)
    )

    assert kwargs["return_frames"] is True
    assert kwargs["return_trajectory_latents"] is False
    assert kwargs["return_trajectory_decoded"] is False
