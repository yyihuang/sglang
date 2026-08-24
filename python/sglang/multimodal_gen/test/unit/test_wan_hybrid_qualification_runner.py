# SPDX-License-Identifier: Apache-2.0

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.qualification.attention_backend_identity import (
    collect_runtime_attention_backend_identity,
)
from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    MODEL_QUALIFICATION_THRESHOLDS,
    PRODUCTION_FA4_BACKEND_CLASS,
    PRODUCTION_FA4_IMPL_CLASS,
    QUALIFICATION_RUN_ORDERS,
    WAN_HYBRID_PROMOTION_GENERATION_HITS,
    WAN_HYBRID_PROMOTION_LAYER_INDICES,
    WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
    _module_git_revision,
    validate_reference_attention_backend_identity,
)
from sglang.multimodal_gen.tools.run_wan_hybrid_qualification import (
    WAN_HYBRID_DEFAULT_LAYER_INDICES,
    WanQualificationConfig,
    _parse_args,
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


def _reference_attention_backend_identity() -> dict:
    return {
        "requested_backend": "fa",
        "resolved_backend_class": PRODUCTION_FA4_BACKEND_CLASS,
        "resolved_impl_class": PRODUCTION_FA4_IMPL_CLASS,
        "implementation": "FA4",
        "flash_attention_version": 4,
        "runtime_observed": True,
        "expected_instance_count": 80,
        "observed_instance_count": 80,
    }


def _coverage(
    scenario: str, num_steps: int = 12, request_id: str = "qualification-request"
) -> dict:
    expected_layers = list(range(40))
    steps = []
    expected_hits = 0
    route_events = 0
    for step_index in range(num_steps):
        component = "transformer" if step_index < num_steps // 2 else "transformer_2"
        actual_timestep = (
            WAN_HYBRID_PROMOTION_MAX_TIMESTEP
            if step_index == num_steps - 1
            else 999 - step_index
        )
        branches = []
        for branch_index in range(2):
            if scenario == "generation":
                eligible = (
                    list(WAN_HYBRID_PROMOTION_LAYER_INDICES)
                    if actual_timestep <= WAN_HYBRID_PROMOTION_MAX_TIMESTEP
                    else []
                )
                fallback = [index for index in expected_layers if index not in eligible]
                control = []
            elif scenario == "full-transformer" and component == "transformer":
                eligible = WAN_HYBRID_DEFAULT_LAYER_INDICES
                fallback = [index for index in expected_layers if index not in eligible]
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
                    "planned_hybrid_layer_indices": eligible,
                    "successful_hybrid_layer_indices": eligible,
                    "eligible_hybrid_miss_layer_indices": [],
                    "unexpected_successful_hybrid_layer_indices": [],
                    "configured_fallback_layer_indices": fallback,
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
                "actual_timestep": actual_timestep,
                "active_component": component,
                "executed_cfg_branch_indices": [0, 1],
                "branches": branches,
            }
        )
    return {
        "schema_version": 2,
        "request_id": request_id,
        "expected_hit_count": expected_hits,
        "actual_hit_count": expected_hits,
        "attributed_actual_hit_count": expected_hits,
        "unattributed_actual_hit_count": 0,
        "eligible_hybrid_miss_count": 0,
        "num_route_events": route_events,
        "num_success_events": expected_hits,
        "steps": steps,
    }


def _generation(
    measure_runs: int, *, include_hits: bool, scenario="generation"
) -> dict:
    request_ids = [f"request-{index}" for index in range(measure_runs)]
    result = {
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "per_run_generation_time_s": [1.0] * measure_runs,
        "per_run_total_duration_ms": [900.0] * measure_runs,
        "timer_scope": "complete DiffGenerator.generate call including output materialization",
        "per_run_output_summaries": [_output_summary() for _ in range(measure_runs)],
        "per_run_request_id": request_ids,
    }
    if include_hits:
        coverages = [
            _coverage(scenario, request_id=request_id) for request_id in request_ids
        ]
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
            "flashinfer_revision": "flashinfer-revision",
            "flashinfer_version": "test",
            "flashinfer_public_api": {
                "WanHybridAttentionWorkspace": True,
                "is_wan_hybrid_attention_available": True,
                "wan_hybrid_attention": True,
            },
            "gpu": {"name": "test GPU"},
        },
        "normalized_backend_request": server_kwargs,
        "port_isolation": {
            "master_port": 30000,
            "reference_scheduler_port": 30001,
            "candidate_scheduler_port": 30002,
            "reference_strict_ports": True,
            "candidate_strict_ports": True,
        },
    }


def _correctness_report(run_order: str, measure_runs: int = 5) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    server_kwargs = {
        "reference": {
            "attention_backend": "fa",
            "master_port": 30000,
            "scheduler_port": 30001,
            "strict_ports": True,
        },
        "candidate": {
            "attention_backend": "wan_hybrid",
            "attention_backend_config": json.dumps(
                {
                    "wan_hybrid_layer_indices": WAN_HYBRID_PROMOTION_LAYER_INDICES,
                    "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "master_port": 30000,
            "scheduler_port": 30002,
            "strict_ports": True,
        },
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
        "reference_attention_backend_identity": (
            _reference_attention_backend_identity()
        ),
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
                "expected_hit_counts": [WAN_HYBRID_PROMOTION_GENERATION_HITS]
                * measure_runs,
                "actual_hit_counts": [WAN_HYBRID_PROMOTION_GENERATION_HITS]
                * measure_runs,
            },
        },
    }
    report["provenance"] = _provenance(server_kwargs)
    report["execution_topology"] = {
        "controller_pid": 1234,
        "controller_process_reused": True,
        "variant_worker_process_sets": 2,
        "variant_worker_process_reused": False,
        "variant_worker_lifecycle": (
            "fresh local DiffGenerator scheduler process set per variant and run order"
        ),
        "same_gpu_worker_process": False,
        "same_cuda_stream_proven": False,
        "port_isolation": report["provenance"]["port_isolation"],
    }
    return report


def _performance_report(measure_runs: int = 5) -> dict:
    server_kwargs = {
        "reference": {
            "attention_backend": "fa",
            "master_port": 30000,
            "scheduler_port": 30001,
            "strict_ports": True,
        },
        "candidate": {
            "attention_backend": "wan_hybrid",
            "attention_backend_config": json.dumps(
                {
                    "wan_hybrid_layer_indices": WAN_HYBRID_PROMOTION_LAYER_INDICES,
                    "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "master_port": 30000,
            "scheduler_port": 30002,
            "strict_ports": True,
        },
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
        "reference_attention_backend_identity": (
            _reference_attention_backend_identity()
        ),
        "order_results": {
            run_order: {
                "reference_generation": _generation(measure_runs, include_hits=False),
                "candidate_generation": _generation(measure_runs, include_hits=True),
                "performance": {"wall_median_speedup": 1.01},
            }
            for run_order in ("reference-first", "candidate-first")
        },
        "qualification": {
            "passed": True,
            "failures": [],
            "thresholds": {
                "required_run_orders": list(QUALIFICATION_RUN_ORDERS),
                "warmup_runs_equals": 2,
                "measure_runs_equals": 5,
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
    report["execution_topology"] = {
        "controller_pid": 1234,
        "controller_process_reused": True,
        "variant_worker_process_sets": 4,
        "variant_worker_process_reused": False,
        "variant_worker_lifecycle": (
            "fresh local DiffGenerator scheduler process set per variant and run order"
        ),
        "same_gpu_worker_process": False,
        "same_cuda_stream_proven": False,
        "port_isolation": provenance["port_isolation"],
    }
    return report


def _full_transformer_forward_report(
    component_name: str,
    run_order: str,
    model_root: Path,
    measure_runs: int = 5,
) -> dict:
    cross_pairs = itertools.product(range(measure_runs), range(measure_runs))
    repeat_pairs = list(itertools.combinations(range(measure_runs), 2))
    identity = {
        "class": "test.TinyWanTransformer",
        "num_blocks": 40,
        "config_sha256": "c" * 64,
        "parameter_manifest_sha256": "d" * 64,
        "buffer_manifest_sha256": "e" * 64,
        "parameter_count": 1,
        "buffer_count": 0,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fixed_input = {
        "hidden_states": {
            "kind": "tensor",
            "shape": [1, 2],
            "dtype": "torch.float32",
            "sha256": "f" * 64,
        }
    }
    component_path = model_root / component_name
    config_path = component_path / "config.json"
    capture_request_id = f"capture-{component_name}"
    capture_coordinates = {
        "step_index": 3,
        "actual_timestep": 500,
        "cfg_branch_index": 0,
    }
    capture_sampling = {
        "prompt_sha256": hashlib.sha256(b"qualification prompt").hexdigest(),
        "parameters": {
            "seed": 4254,
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
        },
    }
    capture_sampling_sha256 = hashlib.sha256(
        json.dumps(capture_sampling, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = {
        "schema_version": 1,
        "component_name": component_name,
        "resolved_component_path": str(component_path.resolve()),
        "component_config_files": [
            {
                "path": "config.json",
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        ],
        "fixed_input": fixed_input,
        "fixed_input_sha256": hashlib.sha256(
            json.dumps(fixed_input, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "reference_model": identity,
        "candidate_model": identity,
        "capture_request_id": capture_request_id,
        "capture_coordinates": capture_coordinates,
        "capture_sampling_sha256": capture_sampling_sha256,
    }
    capture_manifest_binding = {
        "schema_version": 1,
        "request_id": capture_request_id,
        "component_name": component_name,
        "capture_coordinates": capture_coordinates,
        "sampling_sha256": capture_sampling_sha256,
        "input_sha256": binding["fixed_input_sha256"],
    }
    binding["capture_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            capture_manifest_binding, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request_ids = {
        variant: [
            f"{component_name}-{run_order}-{variant}-{run_index}"
            for run_index in range(measure_runs)
        ]
        for variant in ("reference", "candidate")
    }

    def direct_coverage(variant: str, request_id: str) -> dict:
        candidate = variant == "candidate"
        layers = list(range(40))
        hybrid_layers = WAN_HYBRID_DEFAULT_LAYER_INDICES if candidate else []
        hits = len(hybrid_layers)
        return {
            "schema_version": 2,
            "request_id": request_id,
            "expected_hit_count": hits,
            "actual_hit_count": hits,
            "attributed_actual_hit_count": hits,
            "unattributed_actual_hit_count": 0,
            "eligible_hybrid_miss_count": 0,
            "num_route_events": 40,
            "num_success_events": hits,
            "steps": [
                {
                    "step_index": capture_coordinates["step_index"],
                    "actual_timestep": capture_coordinates["actual_timestep"],
                    "active_component": component_name,
                    "executed_cfg_branch_indices": [
                        capture_coordinates["cfg_branch_index"]
                    ],
                    "branches": [
                        {
                            "cfg_branch_index": capture_coordinates["cfg_branch_index"],
                            "num_layers": 40,
                            "layer_indices": layers,
                            "eligible_layer_indices": hybrid_layers,
                            "planned_hybrid_layer_indices": hybrid_layers,
                            "successful_hybrid_layer_indices": hybrid_layers,
                            "eligible_hybrid_miss_layer_indices": [],
                            "unexpected_successful_hybrid_layer_indices": [],
                            "configured_fallback_layer_indices": (
                                [
                                    index
                                    for index in layers
                                    if index not in hybrid_layers
                                ]
                                if candidate
                                else []
                            ),
                            "control_layer_indices": [] if candidate else layers,
                            "expected_hit_count": hits,
                            "actual_hit_count": hits,
                        }
                    ],
                }
            ],
        }

    return {
        "comparison_mode": "correctness",
        "run_order": run_order,
        "warmup_runs": 2,
        "measure_runs": measure_runs,
        "component_name": component_name,
        "evidence_binding": binding,
        "invocation_input_sha256": binding["fixed_input_sha256"],
        "num_blocks": 40,
        "candidate_wan_hybrid_layer_indices": WAN_HYBRID_DEFAULT_LAYER_INDICES,
        "candidate_wan_hybrid_eligible_layer_indices": (
            WAN_HYBRID_DEFAULT_LAYER_INDICES
        ),
        "candidate_backend_exercised": True,
        "candidate_backend_expectation": "exercised",
        "candidate_wan_hybrid_min_timestep": None,
        "candidate_wan_hybrid_max_timestep": None,
        "candidate_per_run_wan_hybrid_hit_count": [1] * measure_runs,
        "candidate_per_run_wan_hybrid_expected_hit_count": [1] * measure_runs,
        "reference_per_run_request_id": request_ids["reference"],
        "candidate_per_run_request_id": request_ids["candidate"],
        "reference_per_run_wan_hybrid_coverage": [
            direct_coverage("reference", request_id)
            for request_id in request_ids["reference"]
        ],
        "candidate_per_run_wan_hybrid_coverage": [
            direct_coverage("candidate", request_id)
            for request_id in request_ids["candidate"]
        ],
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
                "expected_hit_counts": [1] * measure_runs,
                "actual_hit_counts": [1] * measure_runs,
                "candidate_backend_exercised": True,
            },
            "request_local_backend_coverage": {
                "passed": True,
                "failures": [],
            },
        },
    }


def test_plan_uses_generation_invocations_and_direct_full_transformer_evidence(
    tmp_path,
):
    config = _config(tmp_path)

    plan = build_qualification_plan(config)

    assert len(plan) == 6
    for scenario in ("single-block", "generation"):
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
    port_triples = {
        (
            item.command[item.command.index("--master-port") + 1],
            item.command[item.command.index("--reference-scheduler-port") + 1],
            item.command[item.command.index("--candidate-scheduler-port") + 1],
        )
        for item in plan
    }
    assert len(port_triples) == 6
    assert len({port for triple in port_triples for port in triple}) == 18
    assert all("--reference-attention-backend" in item.command for item in plan)

    single_block = next(item for item in plan if item.scenario == "single-block")
    assert '{"wan_hybrid_layer_indices":[0]}' in single_block.command
    assert not any(item.scenario == "full-transformer" for item in plan)
    generation = next(item for item in plan if item.scenario == "generation")
    assert (
        json.dumps(
            {
                "wan_hybrid_layer_indices": WAN_HYBRID_PROMOTION_LAYER_INDICES,
                "wan_hybrid_max_timestep": WAN_HYBRID_PROMOTION_MAX_TIMESTEP,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        in generation.command
    )


def test_plan_rejects_variant_model_overrides_and_nonfixed_performance_counts(
    tmp_path,
):
    with pytest.raises(ValueError, match="model overrides"):
        build_qualification_plan(
            _config(
                tmp_path,
                extra_compare_args=("--candidate-transformer-path=/tmp/other",),
            )
        )
    with pytest.raises(ValueError, match="exactly warmup=2/measure=5"):
        build_qualification_plan(_config(tmp_path, warmup_runs=3))


def test_runtime_attention_identity_uses_resolved_and_executed_instances():
    class ResolvedBackend:
        pass

    class ExecutedImpl:
        _runtime_observed_flash_attention_version = 4

    layer = SimpleNamespace(
        backend=SimpleNamespace(name="FA"),
        attn_impl=ExecutedImpl(),
        _resolved_attn_backend_cls=ResolvedBackend,
    )
    transformer = SimpleNamespace(modules=lambda: [layer])

    identity = collect_runtime_attention_backend_identity(
        [transformer], requested_backend="fa"
    )

    assert identity == {
        "requested_backend": "fa",
        "resolved_backend_class": (
            f"{ResolvedBackend.__module__}.{ResolvedBackend.__qualname__}"
        ),
        "resolved_impl_class": (
            f"{ExecutedImpl.__module__}.{ExecutedImpl.__qualname__}"
        ),
        "implementation": "FA4",
        "flash_attention_version": 4,
        "runtime_observed": True,
        "expected_instance_count": 1,
        "observed_instance_count": 1,
    }
    del ExecutedImpl._runtime_observed_flash_attention_version
    missing_execution = collect_runtime_attention_backend_identity(
        [transformer], requested_backend="fa"
    )
    assert missing_execution["runtime_observed"] is False
    assert missing_execution["observed_instance_count"] == 0


def test_reference_attention_identity_validation_fails_closed():
    assert (
        validate_reference_attention_backend_identity(
            _reference_attention_backend_identity()
        )
        == []
    )
    missing = validate_reference_attention_backend_identity(None)
    assert missing == ["reference runtime attention backend identity is missing"]
    wrong_version = _reference_attention_backend_identity()
    wrong_version["flash_attention_version"] = 3
    errors = validate_reference_attention_backend_identity(wrong_version)
    assert any("flash_attention_version" in error for error in errors)


def test_dry_run_records_neutral_staging_revisions(tmp_path):
    output_dir = tmp_path / "reports"
    manifest = run_qualification_plan(_config(output_dir), dry_run=True)

    assert manifest["staging"] == {
        "label": "b200-validation",
        "sglang_revision": "sglang-revision",
        "flashinfer_revision": "flashinfer-revision",
    }
    assert len(manifest["invocations"]) == 6
    assert manifest["full_transformer_forward_evidence"] == {
        "required": True,
        "scope": "independent-direct-correctness-and-trajectory-off-performance",
        "expected_run_orders": ["reference-first", "candidate-first"],
        "expected_components": ["transformer", "transformer_2"],
        "report_paths": [],
        "performance_report_paths": [],
        "validation_status": "deferred",
        "validation_errors": [],
    }
    assert all(
        invocation["evidence_scope"].startswith("generation-trajectory-")
        for invocation in manifest["invocations"]
    )
    assert not output_dir.exists()


def _write_component_configs(model_root: Path) -> None:
    for component_name in ("transformer", "transformer_2"):
        component_path = model_root / component_name
        component_path.mkdir(parents=True)
        (component_path / "config.json").write_text(
            json.dumps({"component": component_name, "num_layers": 40}),
            encoding="utf-8",
        )


def test_independent_full_transformer_evidence_requires_both_orders(tmp_path):
    model_root = tmp_path / "wan-model"
    _write_component_configs(model_root)
    reports = [
        _full_transformer_forward_report(component_name, run_order, model_root)
        for component_name in ("transformer", "transformer_2")
        for run_order in ("reference-first", "candidate-first")
    ]

    assert (
        validate_full_transformer_forward_evidence(
            reports, expected_model_path=model_root
        )
        == []
    )

    reports[1]["run_order"] = "reference-first"
    errors = validate_full_transformer_forward_evidence(
        reports, expected_model_path=model_root
    )

    assert any("both execution orders" in error for error in errors)


def test_independent_full_transformer_evidence_rejects_relabelled_report(tmp_path):
    model_root = tmp_path / "wan-model"
    _write_component_configs(model_root)
    reports = [
        _full_transformer_forward_report(component_name, run_order, model_root)
        for component_name in ("transformer", "transformer_2")
        for run_order in ("reference-first", "candidate-first")
    ]
    reports[0]["component_name"] = "transformer_2"

    errors = validate_full_transformer_forward_evidence(
        reports, expected_model_path=model_root
    )

    assert any("binding component does not match" in error for error in errors)
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
    assert evidence["observed_performance_reports"] == 0
    assert any(
        "trajectory-off dual-order report" in error
        for error in evidence["validation_errors"]
    )
    assert manifest["invocations"] == []


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


def test_generation_promotion_rejects_route_port_and_topology_tampering(tmp_path):
    config = _config(
        tmp_path,
        scenarios=("generation",),
        modes=("correctness",),
    )
    invocation = build_qualification_plan(config)[0]
    report = _correctness_report(invocation.run_order)
    assert validate_qualification_report(report, invocation, config) == []

    report["server_kwargs"]["candidate"][
        "attention_backend_config"
    ] = '{"wan_hybrid_layer_indices":[39],"wan_hybrid_max_timestep":521}'
    report["server_kwargs"]["reference"]["transformer_weights_path"] = "/tmp/other"
    report["execution_topology"]["same_cuda_stream_proven"] = True
    report["provenance"]["port_isolation"]["candidate_scheduler_port"] = 30001
    report["candidate_generation"]["per_run_wan_hybrid_hit_count"][0] = 5
    errors = validate_qualification_report(report, invocation, config)

    assert "generation candidate is not locked to tail5 at t521" in errors
    assert "reference model override is forbidden" in errors
    assert "execution topology does not match isolated worker semantics" in errors
    assert "port provenance does not match the invocation" in errors
    assert any("requires exactly 10 candidate hits" in error for error in errors)


def test_report_provenance_binds_revision_model_and_sampling(tmp_path):
    config = _config(
        tmp_path,
        scenarios=("generation",),
        modes=("correctness",),
    )
    invocation = build_qualification_plan(config)[0]
    report = _correctness_report(invocation.run_order)
    assert validate_qualification_report(report, invocation, config) == []

    report["provenance"]["runtime"]["flashinfer_revision"] = "wrong-revision"
    report["provenance"]["model"]["resolved_path"] = "/models/other"
    report["provenance"]["fixed_input"]["model_path"] = "/models/other"
    report["provenance"]["fixed_input"]["sampling_kwargs"]["width"] = 123
    errors = validate_qualification_report(report, invocation, config)

    assert "runtime provenance is incomplete or inconsistent" in errors
    assert "model provenance is incomplete" in errors
    assert "provenance.fixed_input.model_path is inconsistent" in errors
    assert any("sampling_kwargs" in error for error in errors)


def test_runtime_revision_is_read_from_module_git_checkout(tmp_path):
    repository = tmp_path / "flashinfer-source"
    module_dir = repository / "flashinfer"
    module_dir.mkdir(parents=True)
    module_file = module_dir / "__init__.py"
    module_file.write_text("__version__ = 'test'\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "flashinfer/__init__.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    expected_revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert _module_git_revision(SimpleNamespace(__file__=str(module_file))) == (
        expected_revision
    )


def test_cli_and_readme_require_four_component_order_reports(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "pass exactly four" in help_text
    assert "pass exactly one" in help_text

    readme = Path(__file__).resolve().parents[2] / "README.md"
    documentation = readme.read_text(encoding="utf-8")
    for report_name in (
        "transformer-reference-first.json",
        "transformer-candidate-first.json",
        "transformer-2-reference-first.json",
        "transformer-2-candidate-first.json",
    ):
        assert report_name in documentation


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
