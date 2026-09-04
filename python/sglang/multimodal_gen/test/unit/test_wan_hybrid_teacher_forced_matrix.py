import copy
import json

import pytest

from sglang.multimodal_gen.tools.validate_wan_hybrid_teacher_forced_matrix import (
    WAN_BLOCK_INDICES,
    WAN_TIMESTEPS,
    validate_teacher_forced_matrix,
)


SGLANG_REVISION = "1" * 40
SGLANG_TREE = "2" * 40
FLASHINFER_REVISION = "3" * 40
FLASHINFER_TREE = "4" * 40
MODEL_PATH = "/model"


def _metrics(*, exact: bool) -> dict:
    return {
        "cosine_similarity": 1.0,
        "mae": 0.0 if exact else 0.01,
        "max_abs": 0.0 if exact else 0.5,
        "finite": True,
        "within_tolerance": True,
        "exact_match": exact,
    }


def _comparison(
    *, exact: bool, reference_run_index: int, candidate_run_index: int
) -> dict:
    return {
        "reference_run_index": reference_run_index,
        "candidate_run_index": candidate_run_index,
        "trajectory_metrics": {
            "num_steps": 12,
            "timesteps_available": True,
            "timesteps_finite": True,
            "timesteps_match": True,
            "per_step_metrics": [
                {
                    **_metrics(exact=exact),
                    "step_index": step_index,
                    "reference_timestep": timestep,
                    "candidate_timestep": timestep,
                }
                for step_index, timestep in enumerate(WAN_TIMESTEPS)
            ],
        },
        "output_metrics": {"all_frames_metrics": _metrics(exact=exact)},
    }


def _repeatability_summary() -> dict:
    return {
        "available": True,
        "num_runs": 2,
        "pairing": "all-pairs",
        "num_pairs": 1,
        "comparisons": [
            _comparison(
                exact=True, reference_run_index=0, candidate_run_index=1
            )
        ],
    }


def _report(step_index: int, timestep: int) -> dict:
    records = [
        {
            "block_index": block_index,
            "timestep": float(timestep),
            "actual_timestep": float(timestep),
            "denoising_step_index": step_index,
            "cfg_negative": cfg_negative,
            "attention_output": _metrics(exact=False),
            "post_residual": _metrics(exact=False),
            "candidate_repeatability": {
                "attention_output": _metrics(exact=True),
                "post_residual": _metrics(exact=True),
            },
        }
        for block_index in WAN_BLOCK_INDICES
        for cfg_negative in (False, True)
    ]
    return {
        "schema_version": 2,
        "model_path": MODEL_PATH,
        "prompt": "fixed prompt",
        "seed": 42,
        "warmup_runs": 0,
        "measure_runs": 2,
        "comparison_mode": "correctness",
        "run_order": "reference-first",
        "trajectory_capture_enabled": True,
        "source_identity": {
            "sglang": {
                "git_revision": SGLANG_REVISION,
                "git_tree": SGLANG_TREE,
                "git_clean": True,
            },
            "flashinfer": {
                "git_revision": FLASHINFER_REVISION,
                "git_tree": FLASHINFER_TREE,
                "git_clean": True,
            },
        },
        "device_identity": {
            "name": "NVIDIA B200",
            "compute_capability": [10, 0],
            "uuid": "GPU-fixed",
        },
        "server_kwargs": {
            "reference": {
                "attention_backend": "fa",
                "enable_cfg_parallel": False,
                "num_gpus": 1,
            },
            "candidate": {
                "attention_backend": "wan_hybrid",
                "enable_cfg_parallel": False,
                "num_gpus": 1,
                "attention_backend_config": json.dumps(
                    {
                        "wan_hybrid_layer_indices": list(WAN_BLOCK_INDICES),
                        "wan_hybrid_min_timestep": timestep,
                        "wan_hybrid_teacher_forced_compare": True,
                        "wan_hybrid_teacher_forced_timestep": timestep,
                    }
                ),
            },
        },
        "sampling_kwargs": {
            "width": 640,
            "height": 384,
            "num_frames": 17,
            "num_inference_steps": 12,
            "guidance_scale": 5.0,
            "return_trajectory_latents": True,
            "return_trajectory_decoded": False,
        },
        "reference_generation": {"per_run_wan_hybrid_hit_count": [0, 0]},
        "candidate_generation": {
            "per_run_wan_hybrid_hit_count": [160, 160],
            "per_run_wan_hybrid_teacher_forced_blocks": [
                copy.deepcopy(records),
                copy.deepcopy(records),
            ],
        },
        "repeatability": {
            "reference": _repeatability_summary(),
            "candidate": _repeatability_summary(),
        },
        "cross_variant_metrics": {
            "reference_num_runs": 2,
            "candidate_num_runs": 2,
            "pairing": "cross-product",
            "num_pairs": 4,
            "comparisons": [
                _comparison(
                    exact=False,
                    reference_run_index=reference_run_index,
                    candidate_run_index=candidate_run_index,
                )
                for reference_run_index in range(2)
                for candidate_run_index in range(2)
            ],
        },
        "qualification": {"passed": True},
    }


def _matrix() -> list[dict]:
    return [
        _report(step_index, timestep)
        for step_index, timestep in enumerate(WAN_TIMESTEPS)
    ]


def _validate(reports: list[dict]) -> dict:
    return validate_teacher_forced_matrix(
        reports,
        expected_sglang_revision=SGLANG_REVISION,
        expected_sglang_tree=SGLANG_TREE,
        expected_flashinfer_revision=FLASHINFER_REVISION,
        expected_flashinfer_tree=FLASHINFER_TREE,
        expected_model_path=MODEL_PATH,
    )


def test_teacher_forced_matrix_accepts_all_960_cells_and_two_runs():
    result = _validate(_matrix())

    assert result["status"] == "PASS"
    assert result["logical_cells"] == 960
    assert result["record_instances"] == 1920
    assert result["same_variant_pairs"] == {"reference": 12, "candidate": 12}
    assert result["cross_variant_pairs"] == 48


def test_teacher_forced_matrix_rejects_missing_or_duplicate_target():
    reports = _matrix()
    with pytest.raises(ValueError, match="exactly 12 reports"):
        _validate(reports[:-1])

    reports[-1] = copy.deepcopy(reports[0])
    with pytest.raises(ValueError, match="duplicate target"):
        _validate(reports)


def test_teacher_forced_matrix_rejects_zero_or_duplicate_cfg_records():
    reports = _matrix()
    reports[0]["candidate_generation"][
        "per_run_wan_hybrid_teacher_forced_blocks"
    ][0] = []
    with pytest.raises(ValueError, match="expected 80 records"):
        _validate(reports)

    reports = _matrix()
    records = reports[0]["candidate_generation"][
        "per_run_wan_hybrid_teacher_forced_blocks"
    ][0]
    records[-1] = copy.deepcopy(records[0])
    with pytest.raises(ValueError, match="block/CFG coverage mismatch"):
        _validate(reports)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("attention_output", "finite"), False, "non-finite tensor"),
        (("attention_output", "within_tolerance"), False, "outside atol/rtol"),
        (
            ("candidate_repeatability", "attention_output", "exact_match"),
            False,
            "not bitwise repeatable",
        ),
    ],
)
def test_teacher_forced_matrix_rejects_bad_local_metrics(path, value, message):
    reports = _matrix()
    record = reports[0]["candidate_generation"][
        "per_run_wan_hybrid_teacher_forced_blocks"
    ][0][0]
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        _validate(reports)


def test_teacher_forced_matrix_rejects_non_target_hits_and_bad_trajectory_step():
    reports = _matrix()
    reports[0]["candidate_generation"]["per_run_wan_hybrid_hit_count"] = [161, 160]
    with pytest.raises(ValueError, match="routed outside the target"):
        _validate(reports)

    reports = _matrix()
    reports[0]["cross_variant_metrics"]["comparisons"][0][
        "trajectory_metrics"
    ]["per_step_metrics"][1]["candidate_timestep"] = 999
    with pytest.raises(ValueError, match="timestep mismatch"):
        _validate(reports)
