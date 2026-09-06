# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.qualification import (
    wan_transformer_capture as capture_module,
)
from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    CAPTURE_BRANCH_ENV,
    CAPTURE_COMPONENTS_ENV,
    CAPTURE_DIR_ENV,
    CAPTURE_REQUEST_ID_ENV,
    CAPTURE_STEPS_ENV,
    WanTransformerInputCapture,
    _model_identity,
    load_wan_transformer_input_capture,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    _model_identity as qualification_model_identity,
)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(40)])
        self.config = {"num_layers": 40}

    def forward(self, hidden_states, timestep=None, guidance=None):
        return hidden_states + 1


def test_model_identity_uses_order_independent_materialized_weights(monkeypatch):
    model = _Model()
    model.register_parameter("placeholder", nn.Parameter(torch.empty(1)))
    materialized = [
        ("weight_b", torch.empty(2, dtype=torch.bfloat16)),
        ("weight_a", torch.empty(3, dtype=torch.float32)),
    ]
    monkeypatch.setattr(
        capture_module,
        "iter_materialized_weights",
        lambda _model: iter(materialized),
    )

    identity = _model_identity(model)
    monkeypatch.setattr(
        capture_module,
        "iter_materialized_weights",
        lambda _model: iter(reversed(materialized)),
    )
    reversed_identity = _model_identity(model)
    assert identity["parameter_count"] == 5
    assert (
        identity["parameter_manifest_sha256"]
        == reversed_identity["parameter_manifest_sha256"]
    )
    assert qualification_model_identity(model) == identity


def _batch(request_id: str):
    sampling = SimpleNamespace(
        prompt="a raccoon",
        seed=0,
        width=640,
        height=384,
        num_frames=17,
        num_inference_steps=12,
        guidance_scale=4.0,
        guidance_scale_2=3.0,
        boundary_ratio=None,
    )
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=sampling,
        is_warmup=False,
    )


def test_worker_capture_round_trips_exact_call_kwargs_and_manifest(tmp_path):
    request_id = "request-a"
    component_path = tmp_path / "model" / "transformer"
    component_path.mkdir(parents=True)
    (component_path / "config.json").write_text('{"num_layers":40}', encoding="utf-8")
    capture = WanTransformerInputCapture(
        output_dir=tmp_path / "capture",
        request_id=request_id,
        components=frozenset({"transformer"}),
    )
    capture.output_dir.mkdir()
    hidden_states = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    batch = _batch(request_id)
    context = ForwardContext(
        current_timestep=2,
        attn_metadata=None,
        forward_batch=batch,
        wan_component_name="transformer",
        wan_actual_timestep=750,
        wan_cfg_branch_index=0,
    )

    manifest_path = capture.capture(
        current_model=_Model(),
        call_kwargs={
            "hidden_states": hidden_states,
            "timestep": torch.tensor([750.0]),
            "nested": (True, {"scale": 4.0}),
        },
        component_name="transformer",
        component_model_path=component_path,
        model_root=component_path.parent,
        forward_context=context,
    )

    call_kwargs, manifest = load_wan_transformer_input_capture(manifest_path)
    assert torch.equal(call_kwargs["hidden_states"], hidden_states)
    assert call_kwargs["hidden_states"].dtype == hidden_states.dtype
    assert call_kwargs["hidden_states"].stride() == hidden_states.stride()
    assert manifest["request_id"] == request_id
    assert manifest["capture"] == {
        "step_index": 2,
        "actual_timestep": 750,
        "cfg_branch_index": 0,
    }
    assert manifest["input"]["hidden_states"]["stride"] == list(hidden_states.stride())
    assert len(manifest["input"]["hidden_states"]["sha256"]) == 64
    assert len(manifest["artifact"]["sha256"]) == 64
    assert json.loads(manifest_path.read_text()) == manifest


def test_worker_capture_records_every_selected_step_and_branch(tmp_path, monkeypatch):
    request_id = "request-all-steps"
    component_path = tmp_path / "model" / "transformer"
    component_path.mkdir(parents=True)
    (component_path / "config.json").write_text('{"num_layers":40}', encoding="utf-8")
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv(CAPTURE_DIR_ENV, str(capture_dir))
    monkeypatch.setenv(CAPTURE_REQUEST_ID_ENV, request_id)
    monkeypatch.setenv(CAPTURE_COMPONENTS_ENV, "transformer")
    monkeypatch.setenv(CAPTURE_BRANCH_ENV, "0,1")
    monkeypatch.setenv(CAPTURE_STEPS_ENV, "*")
    capture = WanTransformerInputCapture.from_environment()
    assert capture is not None
    assert capture.capture_all_steps is True
    assert capture.cfg_branch_indices == frozenset({0, 1})
    batch = _batch(request_id)

    manifest_paths = []
    for step_index in range(2):
        for branch_index in range(2):
            manifest_paths.append(
                capture.capture(
                    current_model=_Model(),
                    call_kwargs={
                        "hidden_states": torch.tensor(
                            [step_index, branch_index], dtype=torch.bfloat16
                        )
                    },
                    component_name="transformer",
                    component_model_path=component_path,
                    model_root=component_path.parent,
                    forward_context=ForwardContext(
                        current_timestep=step_index,
                        attn_metadata=None,
                        forward_batch=batch,
                        wan_component_name="transformer",
                        wan_actual_timestep=999 - step_index,
                        wan_cfg_branch_index=branch_index,
                    ),
                )
            )

    assert all(path is not None for path in manifest_paths)
    assert len(list(capture_dir.glob("*.manifest.json"))) == 4
    coordinates = {
        tuple(
            load_wan_transformer_input_capture(path)[1]["capture"][name]
            for name in ("step_index", "cfg_branch_index")
        )
        for path in manifest_paths
    }
    assert coordinates == {(0, 0), (0, 1), (1, 0), (1, 1)}
    duplicate = capture.capture(
        current_model=_Model(),
        call_kwargs={"hidden_states": torch.ones(1)},
        component_name="transformer",
        component_model_path=component_path,
        model_root=component_path.parent,
        forward_context=ForwardContext(
            current_timestep=1,
            attn_metadata=None,
            forward_batch=batch,
            wan_component_name="transformer",
            wan_actual_timestep=998,
            wan_cfg_branch_index=1,
        ),
    )
    assert duplicate is None


def test_predict_noise_captures_complete_kwargs_before_actual_model_call(tmp_path):
    request_id = "request-a"
    component_path = tmp_path / "model" / "transformer"
    component_path.mkdir(parents=True)
    (component_path / "config.json").write_text("{}", encoding="utf-8")
    capture = WanTransformerInputCapture(
        output_dir=tmp_path / "capture",
        request_id=request_id,
        components=frozenset({"transformer"}),
    )
    capture.output_dir.mkdir()

    class OrderedModel(_Model):
        def forward(self, hidden_states, timestep=None, guidance=None):
            assert list(capture.output_dir.glob("*.manifest.json"))
            return hidden_states + 1

    model = OrderedModel()
    stage = object.__new__(DenoisingStage)
    stage.transformer = model
    stage.transformer_2 = None
    stage._wan_transformer_input_capture = capture
    stage._extra_func_kwarg_names_cache = {}
    stage._bcg_runners = {}
    stage.pipeline = lambda: SimpleNamespace(model_path=str(component_path.parent))
    stage.server_args = SimpleNamespace(component_paths={})
    stage._component_name_for_stage_module = lambda module, default: default
    stage._maybe_get_bcg_runner = lambda module: None
    batch = _batch(request_id)
    hidden_states = torch.tensor([[1.0, 2.0]])

    with set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=batch,
        wan_component_name="transformer",
        wan_actual_timestep=999,
        wan_cfg_branch_index=0,
    ):
        output = stage._predict_noise(
            current_model=model,
            latent_model_input=hidden_states,
            timestep=torch.tensor([999.0]),
            target_dtype=torch.float32,
            guidance=torch.tensor([4.0]),
        )
    assert torch.equal(output, hidden_states + 1)
    manifests = list(capture.output_dir.glob("*.manifest.json"))
    assert len(manifests) == 1
    call_kwargs, _ = load_wan_transformer_input_capture(manifests[0])
    assert set(call_kwargs) == {"hidden_states", "timestep", "guidance"}


def test_capture_rejects_a_second_request_in_singleton_scope(tmp_path):
    capture = WanTransformerInputCapture(
        output_dir=tmp_path,
        request_id="request-a",
        components=frozenset({"transformer"}),
    )
    context = ForwardContext(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=_batch("request-b"),
        wan_component_name="transformer",
        wan_actual_timestep=999,
        wan_cfg_branch_index=0,
    )
    try:
        capture.capture(
            current_model=_Model(),
            call_kwargs={"hidden_states": torch.ones(1)},
            component_name="transformer",
            component_model_path=tmp_path,
            model_root=tmp_path,
            forward_context=context,
        )
    except RuntimeError as error:
        assert "singleton" in str(error)
    else:
        raise AssertionError("capture accepted a second request_id")


def test_capture_loader_rejects_component_changed_after_capture(tmp_path):
    request_id = "request-a"
    component_path = tmp_path / "model" / "transformer"
    component_path.mkdir(parents=True)
    config_path = component_path / "config.json"
    config_path.write_text('{"num_layers":40}', encoding="utf-8")
    capture = WanTransformerInputCapture(
        output_dir=tmp_path / "capture",
        request_id=request_id,
        components=frozenset({"transformer"}),
    )
    capture.output_dir.mkdir()
    manifest_path = capture.capture(
        current_model=_Model(),
        call_kwargs={"hidden_states": torch.ones(1)},
        component_name="transformer",
        component_model_path=component_path,
        model_root=component_path.parent,
        forward_context=ForwardContext(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=_batch(request_id),
            wan_component_name="transformer",
            wan_actual_timestep=999,
            wan_cfg_branch_index=0,
        ),
    )

    config_path.write_text('{"num_layers":39}', encoding="utf-8")
    try:
        load_wan_transformer_input_capture(manifest_path)
    except ValueError as error:
        assert "component file SHA256" in str(error)
    else:
        raise AssertionError("capture loader accepted a changed component config")
