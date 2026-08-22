# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.attention.backends.wan_hybrid import (
    _record_successful_wan_hybrid_forward,
    record_wan_attention_route,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    WanTransformerInputCapture,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    _model_device,
    _move_fixed_input_to_model,
    _warmup_wan_transformer_forward,
    build_wan_transformer_evidence_binding,
    capture_wan_transformer_forward,
    run_wan_transformer_forward_qualification,
    validate_wan_transformer_forward_report,
)
from sglang.multimodal_gen.tools.run_wan_transformer_forward_report import (
    _configure_standalone_layerwise_offload,
)


class _AddBlock(nn.Module):
    def __init__(
        self,
        value: float,
        layer_index: int,
        hybrid_configured: bool,
        eligible_for_hybrid: bool,
    ):
        super().__init__()
        self.value = value
        self.layer_index = layer_index
        self.hybrid_configured = hybrid_configured
        self.eligible_for_hybrid = eligible_for_hybrid

    def forward(self, hidden_states):
        record_wan_attention_route(
            layer_index=self.layer_index,
            hybrid_configured=self.hybrid_configured,
            eligible_for_hybrid=self.eligible_for_hybrid,
        )
        output = hidden_states + self.value
        if self.eligible_for_hybrid:
            output = _record_successful_wan_hybrid_forward(
                output, layer_index=self.layer_index
            )
        return output


class _TinyWanTransformer(nn.Module):
    def __init__(
        self,
        *,
        output_offset: float = 0.0,
        skip_last: bool = False,
        hybrid: bool = False,
        hybrid_layer_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        selected_hybrid_layers = (
            frozenset({39})
            if hybrid and hybrid_layer_indices is None
            else frozenset(hybrid_layer_indices or ())
        )
        self.blocks = nn.ModuleList(
            [
                _AddBlock(
                    0.01,
                    index,
                    hybrid_configured=hybrid,
                    eligible_for_hybrid=index in selected_hybrid_layers,
                )
                for index in range(40)
            ]
        )
        self.output_offset = output_offset
        self.skip_last = skip_last
        self.hybrid = hybrid
        self.wan_hybrid_layer_indices = selected_hybrid_layers if hybrid else None
        self.config = {"num_layers": 40}
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.input_ids = []
        self.timesteps = []
        self.request_ids = []

    def forward(self, hidden_states, timestep=None):
        assert get_forward_context().forward_batch.enable_sequence_shard is False
        assert get_forward_context().forward_batch.enable_teacache is False
        assert get_forward_context().forward_batch.enable_spectrum is False
        self.input_ids.append(id(hidden_states))
        self.timesteps.append(timestep)
        self.request_ids.append(get_forward_context().forward_batch.request_id)
        blocks = self.blocks[:-1] if self.skip_last else self.blocks
        for block in blocks:
            hidden_states = block(hidden_states)
        return hidden_states + self.output_offset


class _AutocastWanTransformer(_TinyWanTransformer):
    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states, encoder_hidden_states, timestep=None):
        conditioned = self.condition(encoder_hidden_states)
        return super().forward(hidden_states + conditioned, timestep=timestep)


class _LayerwiseCudaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states):
        assert hidden_states.is_cuda
        return hidden_states + self.weight


class _LayerwiseCudaTransformer(nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_LayerwiseCudaBlock() for _ in range(2)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _DirectLayerwiseServerArgs:
    component_residency = None
    layerwise_offload_components = ["dit"]
    pipeline_class_name = None
    pin_cpu_memory = False

    @staticmethod
    def has_layerwise_offload_components():
        return True

    @staticmethod
    def is_arg_explicitly_set(_name):
        return False

    @staticmethod
    def layerwise_tuning_for(_component_name, *, dit_group):
        assert dit_group is True
        return 1, 0, "leading"

    @staticmethod
    def record_component_layerwise_capability(_component_name, *, supported):
        assert supported is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_direct_forward_replays_production_cuda_bf16_autocast():
    model = _AutocastWanTransformer().cuda()
    fixed_input = {
        "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16, device="cuda"),
        "encoder_hidden_states": torch.ones(1, 2, dtype=torch.float32, device="cuda"),
        "timestep": torch.tensor([999], device="cuda"),
    }

    with pytest.raises(RuntimeError, match="same dtype"):
        with torch.autocast(device_type="cuda", enabled=False):
            model(**fixed_input)

    _warmup_wan_transformer_forward(
        model,
        fixed_input=fixed_input,
        request_id="autocast-warmup-request",
        component_name="transformer",
        step_index=0,
        actual_timestep=999,
        cfg_branch_index=0,
    )
    trace = capture_wan_transformer_forward(
        model,
        fixed_input=fixed_input,
        request_id="autocast-request",
        component_name="transformer",
        step_index=0,
        actual_timestep=999,
        cfg_branch_index=0,
    )
    assert len(trace.block_outputs) == 40
    assert torch.isfinite(trace.output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA input")
def test_layerwise_offloaded_model_uses_local_execution_device(monkeypatch):
    model = _TinyWanTransformer()
    assert model._device_anchor.device.type == "cpu"
    expected = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.compare_wan_transformer_forward."
        "is_layerwise_offloaded_module",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.compare_wan_transformer_forward."
        "get_local_torch_device",
        lambda: expected,
    )

    fixed_input = {"hidden_states": torch.ones(1, 2)}
    moved_input = _move_fixed_input_to_model(fixed_input, device=_model_device(model))

    assert moved_input["hidden_states"].device == expected
    assert fixed_input["hidden_states"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA offload")
def test_standalone_loader_replays_worker_layerwise_setup():
    model = _LayerwiseCudaTransformer()
    assert next(model.parameters()).device.type == "cpu"

    _configure_standalone_layerwise_offload(
        model,
        component_name="transformer",
        server_args=_DirectLayerwiseServerArgs(),
    )

    assert is_layerwise_offloaded_module(model)
    fixed_input = {"hidden_states": torch.ones(1, 2)}
    moved_input = _move_fixed_input_to_model(fixed_input, device=_model_device(model))
    output = model(**moved_input)
    assert moved_input["hidden_states"].is_cuda
    assert output.is_cuda
    assert torch.equal(output.cpu(), torch.full((1, 2), 3.0))


def _component_path(tmp_path, component_name: str):
    path = tmp_path / "wan-model" / component_name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"num_layers": 40}', encoding="utf-8")
    return path


def _capture_manifest(tmp_path, component_name: str, hidden_states: torch.Tensor):
    request_id = "capture-request"
    component_path = _component_path(tmp_path, component_name)
    batch = SimpleNamespace(
        request_id=request_id,
        is_warmup=False,
        sampling_params=SimpleNamespace(
            prompt="a raccoon",
            seed=0,
            width=640,
            height=384,
            num_frames=17,
            num_inference_steps=12,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            boundary_ratio=None,
        ),
    )
    capture = WanTransformerInputCapture(
        output_dir=tmp_path / f"capture-{component_name}",
        request_id=request_id,
        components=frozenset({component_name}),
    )
    capture.output_dir.mkdir()
    return capture.capture(
        current_model=_TinyWanTransformer(),
        call_kwargs={"hidden_states": hidden_states, "timestep": 500},
        component_name=component_name,
        component_model_path=component_path,
        model_root=component_path.parent,
        forward_context=ForwardContext(
            current_timestep=3,
            attn_metadata=None,
            forward_batch=batch,
            wan_component_name=component_name,
            wan_actual_timestep=500,
            wan_cfg_branch_index=0,
        ),
    )


def test_full_transformer_forward_uses_every_pair_and_every_block(tmp_path):
    reference = _TinyWanTransformer()
    candidate = _TinyWanTransformer(hybrid=True)
    hidden_states = torch.tensor([[1.0, 2.0]])
    manifest = _capture_manifest(tmp_path, "transformer_2", hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        capture_manifest_path=manifest,
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
    assert (
        report["invocation_input_sha256"]
        == report["evidence_binding"]["fixed_input_sha256"]
    )
    assert report["num_blocks"] == 40
    assert report["candidate_wan_hybrid_layer_indices"] == [39]
    assert report["candidate_per_run_wan_hybrid_hit_count"] == [1] * 5
    assert report["candidate_per_run_wan_hybrid_expected_hit_count"] == [1] * 5
    assert len(set(report["reference_per_run_request_id"])) == 5
    assert len(set(report["candidate_per_run_request_id"])) == 5
    assert not set(report["reference_per_run_request_id"]) & set(
        report["candidate_per_run_request_id"]
    )
    assert all(
        coverage["expected_hit_count"] == 0
        and coverage["actual_hit_count"] == 0
        and coverage["num_route_events"] == 40
        and coverage["num_success_events"] == 0
        for coverage in report["reference_per_run_wan_hybrid_coverage"]
    )
    assert all(
        coverage["expected_hit_count"] == 1
        and coverage["actual_hit_count"] == 1
        and coverage["attributed_actual_hit_count"] == 1
        and coverage["unattributed_actual_hit_count"] == 0
        and coverage["eligible_hybrid_miss_count"] == 0
        for coverage in report["candidate_per_run_wan_hybrid_coverage"]
    )
    assert len(set(reference.input_ids)) == 1
    assert set(reference.input_ids) == set(candidate.input_ids)
    assert set(reference.timesteps) == {500}
    assert set(candidate.timesteps) == {500}
    assert len(reference.request_ids) == 7
    assert len(set(reference.request_ids)) == 7
    assert len(candidate.request_ids) == 7
    assert len(set(candidate.request_ids)) == 7
    assert not set(reference.request_ids) & set(candidate.request_ids)

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
    report["candidate_per_run_wan_hybrid_coverage"][0]["steps"][0][
        "actual_timestep"
    ] = 501
    errors = validate_wan_transformer_forward_report(report)

    assert any("run-pair coverage" in error for error in errors)
    assert (
        "every measured candidate hit count must equal the configured depth" in errors
    )
    assert any("capture coordinates" in error for error in errors)


def test_full_transformer_forward_preserves_explicit_multi_layer_override(tmp_path):
    reference = _TinyWanTransformer()
    candidate = _TinyWanTransformer(hybrid=True, hybrid_layer_indices=(35, 39))
    hidden_states = torch.tensor([[1.0, 2.0]])
    manifest = _capture_manifest(tmp_path, "transformer", hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        capture_manifest_path=manifest,
    )

    assert report["candidate_wan_hybrid_layer_indices"] == [35, 39]
    assert report["candidate_per_run_wan_hybrid_hit_count"] == [2] * 5
    assert report["candidate_per_run_wan_hybrid_expected_hit_count"] == [2] * 5
    assert validate_wan_transformer_forward_report(report) == []


def test_full_transformer_report_cannot_be_relabelled(tmp_path):
    reference = _TinyWanTransformer()
    candidate = _TinyWanTransformer(hybrid=True)
    hidden_states = torch.tensor([[1.0, 2.0]])
    manifest = _capture_manifest(tmp_path, "transformer", hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        capture_manifest_path=manifest,
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
            capture_manifest_path=manifest,
        )
    except TypeError as error:
        assert "reference_forward" in str(error)
    else:
        raise AssertionError("caller-owned forward closure was accepted")


def test_full_transformer_forward_checks_final_output_quality(tmp_path):
    reference = _TinyWanTransformer()
    candidate = _TinyWanTransformer(output_offset=2.0, hybrid=True)
    hidden_states = torch.tensor([[1.0, 2.0]])
    manifest = _capture_manifest(tmp_path, "transformer", hidden_states)

    report = run_wan_transformer_forward_qualification(
        reference_model=reference,
        candidate_model=candidate,
        capture_manifest_path=manifest,
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
            model,
            fixed_input={"hidden_states": torch.tensor([[1.0, 2.0]])},
            request_id="request-a",
            component_name="transformer",
            step_index=0,
            actual_timestep=500,
            cfg_branch_index=0,
        )
    except RuntimeError as error:
        assert "captured 39 of 40" in str(error)
    else:
        raise AssertionError("missing full-block coverage did not fail closed")

    assert all(not block._forward_hooks for block in model.blocks)
