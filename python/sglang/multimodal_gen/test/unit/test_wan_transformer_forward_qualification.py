# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
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


class _FakeWanTransformer(nn.Module):
    def __init__(self, *, output_offset: float = 0.0, skip_last: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([_AddBlock(1.0), _AddBlock(2.0), _AddBlock(3.0)])
        self.output_offset = output_offset
        self.skip_last = skip_last

    def forward(self, hidden_states):
        blocks = self.blocks[:-1] if self.skip_last else self.blocks
        for block in blocks:
            hidden_states = block(hidden_states)
        return hidden_states + self.output_offset


def test_full_transformer_forward_uses_every_pair_and_every_block():
    reference = _FakeWanTransformer()
    candidate = _FakeWanTransformer()
    hidden_states = torch.tensor([[1.0, 2.0]])
    candidate_hits = {"count": 0}

    def candidate_forward():
        candidate_hits["count"] += len(candidate.blocks)
        return candidate(hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        reference_forward=lambda: reference(hidden_states),
        candidate_forward=candidate_forward,
        reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
        read_candidate_hit_count=lambda: candidate_hits["count"],
        run_order="candidate-first",
    )

    assert report["qualification"]["passed"] is True
    assert report["qualification"]["failures"] == []
    assert report["run_order"] == "candidate-first"
    assert report["num_blocks"] == 3
    assert report["candidate_per_run_wan_hybrid_hit_count"] == [3] * 5

    cross = report["cross_variant_metrics"]
    assert cross["pairing"] == "cross-product"
    assert cross["num_pairs"] == 25
    assert all(
        comparison["trajectory_metrics"]["num_steps"] == 3
        and len(comparison["trajectory_metrics"]["per_step_metrics"]) == 3
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
    assert "every measured candidate hit count must be positive" in errors


def test_full_transformer_forward_checks_final_output_quality():
    reference = _FakeWanTransformer()
    candidate = _FakeWanTransformer(output_offset=2.0)
    hidden_states = torch.tensor([[1.0, 2.0]])
    candidate_hits = {"count": 0}

    def candidate_forward():
        candidate_hits["count"] += 1
        return candidate(hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        reference_forward=lambda: reference(hidden_states),
        candidate_forward=candidate_forward,
        reset_candidate_hit_count=lambda: candidate_hits.update(count=0),
        read_candidate_hit_count=lambda: candidate_hits["count"],
    )

    assert report["qualification"]["passed"] is False
    assert any(
        failure.get("location") == "transformer_output"
        and failure["reason"] == "mae_above_maximum"
        for failure in report["qualification"]["failures"]
    )


def test_capture_fails_when_forward_skips_a_transformer_block():
    model = _FakeWanTransformer(skip_last=True)

    try:
        capture_wan_transformer_forward(
            model, lambda: model(torch.tensor([[1.0, 2.0]]))
        )
    except RuntimeError as error:
        assert "captured 2 of 3" in str(error)
    else:
        raise AssertionError("missing full-block coverage did not fail closed")

    assert all(not block._forward_hooks for block in model.blocks)
