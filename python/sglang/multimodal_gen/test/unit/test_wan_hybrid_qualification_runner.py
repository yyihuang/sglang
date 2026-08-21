# SPDX-License-Identifier: Apache-2.0

import itertools
import hashlib
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


def _output_summary() -> dict:
    return {
        "sha256": "a" * 64,
        "num_frames": 17,
        "frame_shapes": [[384, 640, 3]] * 17,
        "frame_dtypes": ["uint8"] * 17,
        "finite": True,
    }


def _coverage(scenario: str, num_steps: int = 12) -> dict:
    expected_layers = list(range(40))
    steps = []
    expected_hits = 0
    route_events = 0
    for step_index in range(num_steps):
        component = "transformer" if step_index < num_steps // 2 else "transformer_2"
        branches = []
        for branch_index in range(2):
            if scenario == "generation" or (
                scenario == "full-transformer" and component == "transformer"
            ):
                eligible = expected_layers
                fallback = []
                control = []
            elif scenario == "single-block":
                eligible = [0]
                fallback = expected_layers[1:]
                control = []
            else:
                eligible = []
                fallback = []
                control = expected_layers
            branches.append(
                {
                    "cfg_branch_index": branch_index,
                    "num_layers": 40,
                    "layer_indices": expected_layers,
                    "eligible_layer_indices": eligible,
                    "hybrid_layer_indices": eligible,
                    "successful_hybrid_layer_indices": eligible,
                    "fallback_layer_indices": fallback,
                    "control_layer_indices": control,
                    "expected_hit_count": len(eligible),
                    "actual_hit_count": len(eligible),
                }
            )
            expected_hits += len(eligible)
            route_events += 40
        steps.append(
            {
                "step_index": step_index,
                "actual_timestep": 999 - step_index,
                "active_component": component,
                "executed_cfg_branch_indices": [0, 1],
                "branches": branches,
            }
        )
    return {
        "schema_version": 1,
        "expected_hit_count": expected_hits,
        "actual_hit_count": expected_hits,
        "attributed_actual_hit_count": expected_hits,
        "unattributed_actual_hit_count": 0,
        "eligible_self_fallback_count": 0,
        "num_route_events": route_events,
        "num_success_events": expected_hits,
        "steps": steps,
    }


def _generation(measure_runs: int, *, include_hits: bool, scenario="generation") -> dict:
    result = {
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "per_run_generation_time_s": [1.0] * measure_runs,
        "per_run_total_duration_ms": [900.0] * measure_runs,
        "timer_scope": "complete DiffGenerator.generate call including output materialization",
        "per_run_output_summaries": [_output_summary() for _ in range(measure_runs)],
    }
    if include_hits:
        coverages = [_coverage(scenario) for _ in range(measure_runs)]
        expected = [coverage["expected_hit_count"] for coverage in coverages]
        result["per_run_wan_hybrid_hit_count"] = expected
        result["per_run_wan_hybrid_expected_hit_count"] = expected
        result["per_run_wan_hybrid_coverage"] = coverages
    return result


def _trajectory_comparison(
    reference_index: int, candidate_index: int, num_steps: int = 12
) -> dict:
    return {
        "reference_run_index": reference_index,
        "candidate_run_index": candidate_index,
        "trajectory_metrics": {
            "num_steps": num_steps,
            "per_step_metrics": [{} for _ in range(num_steps)],
        },
    }


def _provenance(server_kwargs: dict) -> dict:
    fixed_input = {
        "model_path": "/models/wan",
        "model_id": "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "prompt": "A curious raccoon",
        "seed": 0,
        "sampling_kwargs": {
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "return_trajectory_latents": True,
            "return_trajectory_decoded": False,
        },
    }
    digest = hashlib.sha256(
        json.dumps(fixed_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "input_sha256": digest,
        "fixed_input": fixed_input,
        "model": {
            "resolved_path": "/models/wan",
            "model_id": fixed_input["model_id"],
            "config_files": [{"path": "model_index.json", "sha256": "b" * 64}],
        },
        "runtime": {
            "sglang_revision": "sglang-revision",
            "flashinfer_version": "test",
            "flashinfer_public_api": {
                "WanHybridAttentionWorkspace": True,
                "is_wan_hybrid_attention_available": True,
                "wan_hybrid_attention": True,
            },
            "gpu": {"name": "test GPU"},
        },
        "normalized_backend_request": server_kwargs,
    }


def _correctness_report(run_order: str, measure_runs: int = 5) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    server_kwargs = {
        "reference": {"attention_backend": "fa"},
        "candidate": {"attention_backend": "wan_hybrid"},
    }
    report = {
        "model_id": "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "prompt": "A curious raccoon",
        "seed": 0,
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "sampling_kwargs": {
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "return_trajectory_latents": True,
            "return_trajectory_decoded": False,
        },
        "server_kwargs": server_kwargs,
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
                "thresholds": {
                    "candidate_hit_count_equals_expected": True,
                    "expected_hit_count_min_exclusive": 0,
                },
                "expected_hit_counts": [960] * measure_runs,
                "actual_hit_counts": [960] * measure_runs,
            },
        },
    }
    report["provenance"] = _provenance(server_kwargs)
    return report


def _performance_report(measure_runs: int = 5) -> dict:
    server_kwargs = {
        "reference": {"attention_backend": "fa"},
        "candidate": {"attention_backend": "wan_hybrid"},
    }
    report = {
        "model_id": "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "prompt": "A curious raccoon",
        "seed": 0,
        "comparison_mode": "performance",
        "run_order": "both",
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "sampling_kwargs": {
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "return_trajectory_latents": False,
            "return_trajectory_decoded": False,
        },
        "server_kwargs": server_kwargs,
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
                "candidate_hit_count_equals_expected": True,
                "expected_hit_count_min_exclusive": 0,
            },
        },
    }
    provenance = _provenance(server_kwargs)
    provenance["fixed_input"]["sampling_kwargs"]["return_trajectory_latents"] = False
    provenance["input_sha256"] = hashlib.sha256(
        json.dumps(
            provenance["fixed_input"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    report["provenance"] = provenance
    return report


def _full_transformer_forward_report(
    component_name: str, run_order: str, measure_runs: int = 5
) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "component_name": component_name,
        "num_blocks": 40,
        "candidate_per_run_wan_hybrid_hit_count": [40] * measure_runs,
        "candidate_per_run_wan_hybrid_expected_hit_count": [40] * measure_runs,
        "cross_variant_metrics": {
            "pairing": "cross-product",
            "comparisons": [
                _trajectory_comparison(reference_index, candidate_index, 40)
                for reference_index, candidate_index in cross_pairs
            ],
        },
        "repeatability": {
            variant: {
                "available": True,
                "pairing": "all-pairs",
                "comparisons": [
                    _trajectory_comparison(reference_index, candidate_index, 40)
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
                "thresholds": {
                    "candidate_hit_count_equals_expected": True,
                    "expected_hit_count_min_exclusive": 0,
                },
                "expected_hit_counts": [40] * measure_runs,
                "actual_hit_counts": [40] * measure_runs,
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
        "scope": "independent-single-forward-all-blocks-for-both-transformers",
        "expected_run_orders": ["reference-first", "candidate-first"],
        "expected_components": ["transformer", "transformer_2"],
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
        _full_transformer_forward_report(component_name, run_order)
        for component_name in ("transformer", "transformer_2")
        for run_order in ("reference-first", "candidate-first")
    ]

    assert validate_full_transformer_forward_evidence(reports) == []

    reports[1]["run_order"] = "reference-first"
    errors = validate_full_transformer_forward_evidence(reports)

    assert any("both execution orders" in error for error in errors)


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
    assert evidence["observed_component_run_orders"] == []
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

    assert any("wan_hybrid hit evidence" in error for error in errors)
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
