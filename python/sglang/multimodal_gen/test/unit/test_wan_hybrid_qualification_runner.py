# SPDX-License-Identifier: Apache-2.0

import itertools
import json
from pathlib import Path

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
    QUALIFICATION_RUN_ORDERS,
)
from sglang.multimodal_gen.tools.run_wan_hybrid_qualification import (
    WanQualificationConfig,
    build_qualification_plan,
    run_qualification_plan,
    validate_full_transformer_forward_evidence,
    validate_qualification_report,
)


def _config(tmp_path: Path, **overrides) -> WanQualificationConfig:
    defaults = {
        "model_path": "/models/wan",
        "model_id": "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "output_dir": tmp_path,
        "sglang_revision": "sglang-revision",
        "flashinfer_revision": "flashinfer-revision",
        "staging_label": "b200-validation",
    }
    return WanQualificationConfig(**(defaults | overrides))


def _generation(measure_runs: int, *, include_hits: bool) -> dict:
    result = {
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "per_run_generation_time_s": [1.0] * measure_runs,
    }
    if include_hits:
        result["per_run_wan_hybrid_hit_count"] = [40] * measure_runs
    return result


def _trajectory_comparison(reference_index: int, candidate_index: int) -> dict:
    return {
        "reference_run_index": reference_index,
        "candidate_run_index": candidate_index,
        "trajectory_metrics": {
            "num_steps": 3,
            "per_step_metrics": [{}, {}, {}],
        },
    }


def _correctness_report(run_order: str, measure_runs: int = 5) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "sampling_kwargs": {"return_trajectory_latents": True},
        "server_kwargs": {
            "reference": {"attention_backend": "fa"},
            "candidate": {"attention_backend": "wan_hybrid"},
        },
        "reference_generation": _generation(measure_runs, include_hits=False),
        "candidate_generation": _generation(measure_runs, include_hits=True),
        "cross_variant_metrics": {
            "pairing": "cross-product",
            "comparisons": [
                _trajectory_comparison(reference_index, candidate_index)
                for reference_index, candidate_index in cross_pairs
            ],
        },
        "repeatability": {
            variant: {
                "available": True,
                "pairing": "all-pairs",
                "comparisons": [
                    _trajectory_comparison(reference_index, candidate_index)
                    for reference_index, candidate_index in repeat_pairs
                ],
            }
            for variant in ("reference", "candidate")
        },
        "qualification": {
            "passed": True,
            "failures": [],
            "thresholds": MODEL_QUALIFICATION_THRESHOLDS,
            "candidate_backend_hits": {
                "passed": True,
                "failures": [],
                "thresholds": {"candidate_hit_count_min_exclusive": 0},
            },
        },
    }


def _performance_report(measure_runs: int = 5) -> dict:
    return {
        "comparison_mode": "performance",
        "run_order": "both",
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "sampling_kwargs": {"return_trajectory_latents": False},
        "server_kwargs": {
            "reference": {"attention_backend": "fa"},
            "candidate": {"attention_backend": "wan_hybrid"},
        },
        "order_results": {
            run_order: {
                "reference_generation": _generation(
                    measure_runs, include_hits=False
                ),
                "candidate_generation": _generation(
                    measure_runs, include_hits=True
                ),
                "performance": {"wall_median_speedup": 1.01},
            }
            for run_order in ("reference-first", "candidate-first")
        },
        "qualification": {
            "passed": True,
            "failures": [],
            "thresholds": {
                "required_run_orders": list(QUALIFICATION_RUN_ORDERS),
                "warmup_runs_min": 2,
                "measure_runs_min": 5,
                "speedup_min": 1.0,
                "candidate_hit_count_min_exclusive": 0,
            },
        },
    }


def _full_transformer_forward_report(run_order: str, measure_runs: int = 5) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "num_blocks": 3,
        "candidate_per_run_wan_hybrid_hit_count": [3] * measure_runs,
        "cross_variant_metrics": {
            "pairing": "cross-product",
            "comparisons": [
                _trajectory_comparison(reference_index, candidate_index)
                for reference_index, candidate_index in cross_pairs
            ],
        },
        "repeatability": {
            variant: {
                "available": True,
                "pairing": "all-pairs",
                "comparisons": [
                    _trajectory_comparison(reference_index, candidate_index)
                    for reference_index, candidate_index in repeat_pairs
                ],
            }
            for variant in ("reference", "candidate")
        },
        "qualification": {
            "passed": True,
            "failures": [],
            "thresholds": MODEL_QUALIFICATION_THRESHOLDS,
            "candidate_backend_hits": {
                "passed": True,
                "failures": [],
                "thresholds": {"candidate_hit_count_min_exclusive": 0},
            },
        },
    }


def test_plan_uses_two_correctness_orders_and_one_dual_order_performance(tmp_path):
    config = _config(tmp_path)

    plan = build_qualification_plan(config)

    assert len(plan) == 9
    for scenario in ("single-block", "full-transformer", "generation"):
        scenario_plan = [item for item in plan if item.scenario == scenario]
        assert [item.run_order for item in scenario_plan] == [
            "reference-first",
            "candidate-first",
            "both",
        ]
        assert [item.comparison_mode for item in scenario_plan] == [
            "correctness",
            "correctness",
            "performance",
        ]
    ports = {
        item.command[item.command.index("--master-port") + 1] for item in plan
    }
    assert len(ports) == 9
    assert all("--reference-attention-backend" in item.command for item in plan)

    single_block = next(item for item in plan if item.scenario == "single-block")
    assert '{"wan_hybrid_layer_indices":[0]}' in single_block.command
    full_transformer = next(
        item for item in plan if item.scenario == "full-transformer"
    )
    assert "transformer=wan_hybrid" in full_transformer.command
    assert "transformer_2=fa" in full_transformer.command
    assert full_transformer.evidence_scope == (
        "generation-trajectory-primary-transformer-component"
    )


def test_dry_run_records_neutral_staging_revisions(tmp_path):
    output_dir = tmp_path / "reports"
    manifest = run_qualification_plan(_config(output_dir), dry_run=True)

    assert manifest["staging"] == {
        "label": "b200-validation",
        "sglang_revision": "sglang-revision",
        "flashinfer_revision": "flashinfer-revision",
    }
    assert len(manifest["invocations"]) == 9
    assert manifest["full_transformer_forward_evidence"] == {
        "required": True,
        "scope": "independent-single-forward-all-transformer-blocks",
        "expected_run_orders": ["reference-first", "candidate-first"],
        "report_paths": [],
        "validation_status": "deferred",
        "validation_errors": [],
    }
    assert all(
        invocation["evidence_scope"].startswith("generation-trajectory-")
        for invocation in manifest["invocations"]
    )
    assert not output_dir.exists()


def test_independent_full_transformer_evidence_requires_both_orders():
    reports = [
        _full_transformer_forward_report("reference-first"),
        _full_transformer_forward_report("candidate-first"),
    ]

    assert validate_full_transformer_forward_evidence(reports) == []

    reports[1]["run_order"] = "reference-first"
    errors = validate_full_transformer_forward_evidence(reports)

    assert any("cover both execution orders" in error for error in errors)


def test_full_transformer_plan_fails_before_generation_without_forward_evidence(
    tmp_path,
):
    output_dir = tmp_path / "reports"
    config = _config(
        output_dir,
        scenarios=("full-transformer",),
        modes=("performance",),
    )

    try:
        run_qualification_plan(config)
    except RuntimeError as error:
        assert "full-transformer evidence failed validation" in str(error)
    else:
        raise AssertionError("missing independent forward evidence did not fail")

    manifest = json.loads(
        (output_dir / "wan-hybrid-qualification-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["passed"] is False
    evidence = manifest["full_transformer_forward_evidence"]
    assert evidence["validation_status"] == "failed"
    assert evidence["observed_run_orders"] == []
    assert all(
        invocation["status"] == "not-run-invalid-forward-evidence"
        for invocation in manifest["invocations"]
    )


def test_correctness_gate_requires_new_hit_field_and_complete_pair_step_coverage(
    tmp_path,
):
    config = _config(
        tmp_path,
        scenarios=("generation",),
        modes=("correctness",),
    )
    invocation = build_qualification_plan(config)[0]
    report = _correctness_report(invocation.run_order)

    assert validate_qualification_report(report, invocation, config) == []

    report["candidate_generation"]["per_run_legacy_hit_count"] = [40] * 5
    del report["candidate_generation"]["per_run_wan_hybrid_hit_count"]
    report["cross_variant_metrics"]["comparisons"][0]["trajectory_metrics"][
        "per_step_metrics"
    ].pop()
    errors = validate_qualification_report(report, invocation, config)

    assert any("wan_hybrid hit count" in error for error in errors)
    assert any("all-step coverage" in error for error in errors)


def test_performance_gate_requires_both_orders_without_trajectory(tmp_path):
    config = _config(
        tmp_path,
        scenarios=("generation",),
        modes=("performance",),
    )
    invocation = build_qualification_plan(config)[0]
    report = _performance_report()

    assert validate_qualification_report(report, invocation, config) == []

    report["order_results"]["candidate-first"]["performance"][
        "wall_median_speedup"
    ] = 0.99
    report["sampling_kwargs"]["return_trajectory_latents"] = True
    errors = validate_qualification_report(report, invocation, config)

    assert any("below 1.0" in error for error in errors)
    assert "performance unexpectedly captured trajectory latents" in errors
