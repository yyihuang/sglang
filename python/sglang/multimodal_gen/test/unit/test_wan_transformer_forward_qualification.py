# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    build_wan_transformer_evidence_binding,
    capture_wan_transformer_forward,
    run_wan_transformer_forward_qualification,
    validate_wan_transformer_forward_report,
)


class _AddBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states):
        return hidden_states + self.value


class _TinyWanTransformer(nn.Module):
    def __init__(
        self,
        *,
        output_offset: float = 0.0,
        skip_last: bool = False,
        hit_counter: dict[str, int] | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([_AddBlock(0.01) for _ in range(40)])
        self.output_offset = output_offset
        self.skip_last = skip_last
        self.hit_counter = hit_counter
        self.input_ids = []
        self.timesteps = []

    def forward(self, hidden_states, timestep=None):
        self.input_ids.append(id(hidden_states))
        self.timesteps.append(timestep)
        blocks = self.blocks[:-1] if self.skip_last else self.blocks
        for block in blocks:
            hidden_states = block(hidden_states)
        if self.hit_counter is not None:
            self.hit_counter["count"] += len(blocks)
        return hidden_states + self.output_offset


def _component_path(tmp_path, component_name: str):
    path = tmp_path / "wan-model" / component_name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"num_layers": 40}', encoding="utf-8")
    return path


def test_full_transformer_forward_uses_every_pair_and_every_block(tmp_path):
    reference = _TinyWanTransformer()
    candidate_hits = {"count": 0}
    candidate = _TinyWanTransformer(hit_counter=candidate_hits)
    hidden_states = torch.tensor([[1.0, 2.0]])

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
        read_candidate_hit_count=lambda: candidate_hits["count"],
        component_name="transformer_2",
        component_model_path=_component_path(tmp_path, "transformer_2"),
        fixed_input={"hidden_states": hidden_states, "timestep": 500},
        run_order="candidate-first",
    )

    assert report["qualification"]["passed"] is True
    assert report["qualification"]["failures"] == []
    assert report["run_order"] == "candidate-first"
    assert report["component_name"] == "transformer_2"
    assert report["evidence_binding"]["component_name"] == "transformer_2"
    assert report["evidence_binding"]["fixed_input"]["hidden_states"]["kind"] == (
        "tensor"
    )
    assert report["invocation_input_sha256"] == report["evidence_binding"][
        "fixed_input_sha256"
    ]
    assert report["num_blocks"] == 40
    assert report["candidate_per_run_wan_hybrid_hit_count"] == [40] * 5
    assert report["candidate_per_run_wan_hybrid_expected_hit_count"] == [40] * 5
    assert set(reference.input_ids) == {id(hidden_states)}
    assert set(candidate.input_ids) == {id(hidden_states)}
    assert set(reference.timesteps) == {500}
    assert set(candidate.timesteps) == {500}

    cross = report["cross_variant_metrics"]
    assert cross["pairing"] == "cross-product"
    assert cross["num_pairs"] == 25
    assert all(
        comparison["trajectory_metrics"]["num_steps"] == 40
        and len(comparison["trajectory_metrics"]["per_step_metrics"]) == 40
        for comparison in cross["comparisons"]
    )
    for variant in ("reference", "candidate"):
        repeatability = report["repeatability"][variant]
        assert repeatability["pairing"] == "all-pairs"
        assert repeatability["num_pairs"] == 10

    assert all(not block._forward_hooks for block in reference.blocks)
    assert all(not block._forward_hooks for block in candidate.blocks)
    assert validate_wan_transformer_forward_report(report) == []

    report["cross_variant_metrics"]["comparisons"].pop()
    report["candidate_per_run_wan_hybrid_hit_count"][0] = 0
    errors = validate_wan_transformer_forward_report(report)

    assert any("run-pair coverage" in error for error in errors)
    assert "every measured candidate hit count must equal expected depth 40" in errors


def test_full_transformer_report_cannot_be_relabelled(tmp_path):
    reference = _TinyWanTransformer()
    candidate_hits = {"count": 0}
    candidate = _TinyWanTransformer(hit_counter=candidate_hits)
    hidden_states = torch.tensor([[1.0, 2.0]])

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
        read_candidate_hit_count=lambda: candidate_hits["count"],
        component_name="transformer",
        component_model_path=_component_path(tmp_path, "transformer"),
        fixed_input={"hidden_states": hidden_states, "timestep": 500},
    )
    report["component_name"] = "transformer_2"

    errors = validate_wan_transformer_forward_report(
        report, expected_model_path=tmp_path / "wan-model"
    )

    assert "evidence binding component does not match report" in errors
    assert "resolved component path does not match model/component" in errors

    try:
        run_wan_transformer_forward_qualification(
            reference_model=reference,
            candidate_model=candidate,
            reference_forward=lambda: reference(torch.zeros_like(hidden_states)),
            reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
            read_candidate_hit_count=lambda: candidate_hits["count"],
            component_name="transformer",
            component_model_path=tmp_path / "wan-model" / "transformer",
            fixed_input={"hidden_states": hidden_states},
        )
    except TypeError as error:
        assert "reference_forward" in str(error)
    else:
        raise AssertionError("caller-owned forward closure was accepted")


def test_full_transformer_forward_checks_final_output_quality(tmp_path):
    reference = _TinyWanTransformer()
    candidate_hits = {"count": 0}
    candidate = _TinyWanTransformer(
        output_offset=2.0, hit_counter=candidate_hits
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
        read_candidate_hit_count=lambda: candidate_hits["count"],
        component_name="transformer",
        component_model_path=_component_path(tmp_path, "transformer"),
        fixed_input={"hidden_states": hidden_states, "timestep": 500},
    )

    assert report["qualification"]["passed"] is False
    assert any(
        failure.get("location") == "transformer_output"
        and failure["reason"] == "mae_above_maximum"
        for failure in report["qualification"]["failures"]
    )


def test_component_binding_is_stable_across_fresh_model_instances(tmp_path):
    component_path = _component_path(tmp_path, "transformer")
    hidden_states = torch.tensor([[1.0, 2.0]])
    models = [_TinyWanTransformer() for _ in range(4)]
    for model in models:
        model.config = {"opaque_runtime_value": object(), "num_layers": 40}

    first = build_wan_transformer_evidence_binding(
        component_name="transformer",
        component_model_path=component_path,
        fixed_input={"hidden_states": hidden_states},
        reference_model=models[0],
        candidate_model=models[1],
    )
    second = build_wan_transformer_evidence_binding(
        component_name="transformer",
        component_model_path=component_path,
        fixed_input={"hidden_states": hidden_states},
        reference_model=models[2],
        candidate_model=models[3],
    )

    assert first == second


def test_capture_fails_when_forward_skips_a_transformer_block():
    model = _TinyWanTransformer(skip_last=True)

    try:
        capture_wan_transformer_forward(
            model, lambda: model(torch.tensor([[1.0, 2.0]]))
        )
    except RuntimeError as error:
        assert "captured 39 of 40" in str(error)
    else:
        raise AssertionError("missing full-block coverage did not fail closed")

    assert all(not block._forward_hooks for block in model.blocks)
