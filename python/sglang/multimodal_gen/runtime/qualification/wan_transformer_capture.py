# SPDX-License-Identifier: Apache-2.0

"""Capture the exact kwargs passed to a Wan transformer during serving.

This path is deliberately opt-in.  When the capture directory environment
variable is absent, construction returns ``None`` and the serving path performs
no tensor inspection, copies, hashing, or filesystem work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    iter_materialized_weights,
)

CAPTURE_DIR_ENV = "SGLANG_WAN_TRANSFORMER_INPUT_CAPTURE_DIR"
CAPTURE_COMPONENTS_ENV = "SGLANG_WAN_TRANSFORMER_INPUT_CAPTURE_COMPONENTS"
CAPTURE_BRANCH_ENV = "SGLANG_WAN_TRANSFORMER_INPUT_CAPTURE_CFG_BRANCH"
CAPTURE_STEPS_ENV = "SGLANG_WAN_TRANSFORMER_INPUT_CAPTURE_STEPS"
CAPTURE_REQUEST_ID_ENV = "SGLANG_WAN_TRANSFORMER_INPUT_CAPTURE_REQUEST_ID"
WAN_COMPONENT_NAMES = ("transformer", "transformer_2")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_value(value: Any, *, location: str) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).cpu()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{location} mapping keys must be strings")
        return {
            key: _stable_value(item, location=f"{location}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [
            _stable_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _stable_value(dataclasses.asdict(value), location=location)
    raise TypeError(f"{location} has unsupported capture type {type(value).__name__}")


def _snapshot_value(value: Any, *, location: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{location} must use strided tensor storage")
        detached = value.detach()
        snapshot = torch.empty_strided(
            detached.shape,
            detached.stride(),
            dtype=detached.dtype,
            device="cpu",
        )
        snapshot.copy_(detached, non_blocking=False)
        return snapshot
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{location} mapping keys must be strings")
        return {
            key: _snapshot_value(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _snapshot_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _snapshot_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{location} has unsupported capture type {type(value).__name__}")


def _model_identity(model: torch.nn.Module) -> dict[str, Any]:
    parameters = sorted(
        (
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
            }
            for name, parameter in iter_materialized_weights(model)
        ),
        key=lambda item: item["name"],
    )
    buffers = [
        {
            "name": name,
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype),
            "numel": buffer.numel(),
        }
        for name, buffer in model.named_buffers()
    ]
    config = {}
    for name in ("config", "hf_config"):
        value = getattr(model, name, None)
        if value is None:
            continue
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
            value = vars(value)
        config[name] = _stable_config_value(value)
    identity = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "num_blocks": len(tuple(getattr(model, "blocks", ()))),
        "config_sha256": _sha256_json(config),
        "parameter_manifest_sha256": _sha256_json(parameters),
        "buffer_manifest_sha256": _sha256_json(buffers),
        "parameter_count": sum(item["numel"] for item in parameters),
        "buffer_count": sum(item["numel"] for item in buffers),
    }
    return identity | {"identity_sha256": _sha256_json(identity)}


def _stable_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_config_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_stable_config_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (Path, torch.dtype, torch.device)):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _stable_config_value(dataclasses.asdict(value))
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _sampling_binding(batch: Any) -> dict[str, Any]:
    sampling_params = getattr(batch, "sampling_params", None)
    prompt = getattr(sampling_params, "prompt", None)
    prompt_summary = _stable_value(prompt, location="sampling.prompt")
    selected = {
        name: getattr(sampling_params, name, None)
        for name in (
            "seed",
            "width",
            "height",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "boundary_ratio",
        )
    }
    binding = {
        "prompt_sha256": _sha256_json(prompt_summary),
        "parameters": _stable_value(selected, location="sampling.parameters"),
    }
    return binding | {"sampling_sha256": _sha256_json(binding)}


def _component_config_files(component_path: Path) -> list[dict[str, str]]:
    files = []
    for name in (
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        path = component_path / name
        if path.is_file():
            files.append({"path": name, "sha256": _sha256_file(path)})
    if not any(item["path"] == "config.json" for item in files):
        raise FileNotFoundError(
            f"Wan transformer capture requires component config: "
            f"{component_path / 'config.json'}"
        )
    return files


def _parse_components(raw: str) -> frozenset[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    components = frozenset(items)
    if not components or not components <= set(WAN_COMPONENT_NAMES):
        raise ValueError(
            f"{CAPTURE_COMPONENTS_ENV} must select transformer and/or transformer_2"
        )
    if len(items) != len(components):
        raise ValueError(f"{CAPTURE_COMPONENTS_ENV} contains duplicate components")
    return components


def _parse_steps(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    result = {}
    for item in raw.split(","):
        component, separator, step = item.strip().partition("=")
        if separator != "=" or component not in WAN_COMPONENT_NAMES:
            raise ValueError(f"{CAPTURE_STEPS_ENV} entries must use component=step")
        if component in result:
            raise ValueError(f"{CAPTURE_STEPS_ENV} contains duplicate components")
        try:
            step_index = int(step)
        except ValueError as error:
            raise ValueError(
                f"{CAPTURE_STEPS_ENV} step values must be integers"
            ) from error
        if step_index < 0:
            raise ValueError(f"{CAPTURE_STEPS_ENV} step values must be nonnegative")
        result[component] = step_index
    return result


def _parse_branches(raw: str | None) -> frozenset[int]:
    if not raw:
        return frozenset({0})
    branches = []
    for item in raw.split(","):
        try:
            branch_index = int(item.strip())
        except ValueError as error:
            raise ValueError(
                f"{CAPTURE_BRANCH_ENV} entries must be integers"
            ) from error
        if branch_index < 0:
            raise ValueError(f"{CAPTURE_BRANCH_ENV} entries must be nonnegative")
        branches.append(branch_index)
    result = frozenset(branches)
    if not result or len(result) != len(branches):
        raise ValueError(f"{CAPTURE_BRANCH_ENV} must contain unique branch indices")
    return result


def _safe_request_stem(request_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:48] or "request"
    return f"{safe}-{hashlib.sha256(request_id.encode()).hexdigest()[:12]}"


def load_wan_transformer_input_capture(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify a worker-produced tensor artifact and JSON manifest."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Wan transformer capture schema")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _sha256_json(manifest_payload):
        raise ValueError("Wan transformer capture manifest SHA256 is inconsistent")
    component = manifest.get("component")
    if not isinstance(component, dict):
        raise ValueError("Wan transformer capture component binding is missing")
    component_payload = {
        key: value for key, value in component.items() if key != "component_sha256"
    }
    if component.get("component_sha256") != _sha256_json(component_payload):
        raise ValueError("Wan transformer capture component SHA256 is inconsistent")
    component_path = Path(component.get("resolved_path", "")).expanduser().resolve()
    config_files = component.get("config_files")
    if not isinstance(config_files, list) or not config_files:
        raise ValueError("Wan transformer capture component files are missing")
    for item in config_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Wan transformer capture component file is invalid")
        config_path = component_path / item["path"]
        if not config_path.is_file() or _sha256_file(config_path) != item.get("sha256"):
            raise ValueError(
                "Wan transformer captured component file SHA256 is inconsistent"
            )
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("Wan transformer capture sampling binding is missing")
    sampling_payload = {
        key: value for key, value in sampling.items() if key != "sampling_sha256"
    }
    if sampling.get("sampling_sha256") != _sha256_json(sampling_payload):
        raise ValueError("Wan transformer capture sampling SHA256 is inconsistent")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("Wan transformer capture model binding is missing")
    model_payload = {
        key: value for key, value in model.items() if key != "identity_sha256"
    }
    if model.get("identity_sha256") != _sha256_json(model_payload):
        raise ValueError("Wan transformer capture model SHA256 is inconsistent")
    if not isinstance(manifest.get("request_id"), str) or not manifest["request_id"]:
        raise ValueError("Wan transformer capture request binding is missing")
    coordinates = manifest.get("capture")
    if not isinstance(coordinates, dict) or any(
        isinstance(coordinates.get(name), bool)
        or not isinstance(coordinates.get(name), int)
        for name in ("step_index", "actual_timestep", "cfg_branch_index")
    ):
        raise ValueError("Wan transformer capture coordinates are invalid")
    artifact_path = manifest_path.parent / manifest["artifact"]["path"]
    if _sha256_file(artifact_path) != manifest["artifact"]["sha256"]:
        raise ValueError("Wan transformer tensor artifact SHA256 is inconsistent")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    call_kwargs = payload.get("call_kwargs") if isinstance(payload, dict) else None
    if not isinstance(call_kwargs, dict):
        raise ValueError("Wan transformer tensor artifact has no call_kwargs mapping")
    input_summary = _stable_value(call_kwargs, location="call_kwargs")
    if payload.get("input_sha256") != manifest.get("input_sha256"):
        raise ValueError(
            "Wan transformer tensor payload SHA256 binding is inconsistent"
        )
    if _sha256_json(input_summary) != manifest.get("input_sha256"):
        raise ValueError("Wan transformer captured input SHA256 is inconsistent")
    return call_kwargs, manifest


@dataclass(frozen=True)
class WanTransformerInputCapture:
    output_dir: Path
    request_id: str
    components: frozenset[str] = frozenset(WAN_COMPONENT_NAMES)
    cfg_branch_indices: frozenset[int] = frozenset({0})
    selected_steps: Mapping[str, int] = dataclasses.field(default_factory=dict)
    capture_all_steps: bool = False

    @classmethod
    def from_environment(cls) -> "WanTransformerInputCapture | None":
        raw_output_dir = os.environ.get(CAPTURE_DIR_ENV)
        if raw_output_dir is None:
            return None
        output_dir = Path(raw_output_dir).expanduser().resolve()
        request_id = os.environ.get(CAPTURE_REQUEST_ID_ENV)
        if not request_id:
            raise ValueError(
                f"{CAPTURE_REQUEST_ID_ENV} is required for singleton request capture"
            )
        components = _parse_components(
            os.environ.get(CAPTURE_COMPONENTS_ENV, ",".join(WAN_COMPONENT_NAMES))
        )
        branches = _parse_branches(os.environ.get(CAPTURE_BRANCH_ENV))
        raw_steps = os.environ.get(CAPTURE_STEPS_ENV)
        capture_all_steps = raw_steps == "*"
        selected_steps = {} if capture_all_steps else _parse_steps(raw_steps)
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            output_dir=output_dir,
            request_id=request_id,
            components=components,
            cfg_branch_indices=branches,
            selected_steps=selected_steps,
            capture_all_steps=capture_all_steps,
        )

    def capture(
        self,
        *,
        current_model: torch.nn.Module,
        call_kwargs: Mapping[str, Any],
        component_name: str,
        component_model_path: str | Path,
        model_root: str | Path,
        forward_context: Any,
    ) -> Path | None:
        if component_name not in self.components:
            return None
        if forward_context.wan_component_name != component_name:
            raise RuntimeError(
                "Wan transformer capture component does not match actual model identity"
            )
        branch_index = forward_context.wan_cfg_branch_index
        if branch_index not in self.cfg_branch_indices:
            return None
        step_index = forward_context.current_timestep
        selected_step = self.selected_steps.get(component_name)
        if selected_step is not None and step_index != selected_step:
            return None
        batch = forward_context.forward_batch
        if batch is None:
            raise RuntimeError("Wan transformer capture requires a serving request")
        request_id = getattr(batch, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("Wan transformer capture requires a request_id")
        if request_id != self.request_id:
            raise RuntimeError(
                "Wan transformer capture observed a request outside its singleton "
                "qualification request"
            )
        if getattr(batch, "is_warmup", False):
            return None
        captured = getattr(batch, "_wan_transformer_input_capture_keys", None)
        if captured is None:
            captured = set()
            batch._wan_transformer_input_capture_keys = captured
        capture_key: str | tuple[str, int, int] = component_name
        if self.capture_all_steps:
            capture_key = (component_name, step_index, branch_index)
        if capture_key in captured:
            return None

        snapshot = _snapshot_value(call_kwargs, location="call_kwargs")
        input_summary = _stable_value(snapshot, location="call_kwargs")
        input_sha256 = _sha256_json(input_summary)
        component_path = Path(component_model_path).expanduser().resolve()
        model_root = Path(model_root).expanduser().resolve()
        actual_timestep = forward_context.wan_actual_timestep
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise RuntimeError("Wan transformer capture step index is not an integer")
        if isinstance(actual_timestep, bool) or not isinstance(actual_timestep, int):
            raise RuntimeError("Wan transformer capture timestep is not an integer")

        stem = (
            f"{_safe_request_stem(request_id)}.{component_name}."
            f"step{step_index}.branch{branch_index}"
        )
        artifact_path = self.output_dir / f"{stem}.inputs.pt"
        manifest_path = self.output_dir / f"{stem}.manifest.json"
        sampling = _sampling_binding(batch)
        model = _model_identity(current_model)
        component = {
            "name": component_name,
            "resolved_path": str(component_path),
            "config_files": _component_config_files(component_path),
        }
        component["component_sha256"] = _sha256_json(component)
        payload = {
            "schema_version": 1,
            "call_kwargs": snapshot,
            "input_sha256": input_sha256,
        }
        with tempfile.NamedTemporaryFile(
            dir=self.output_dir, prefix=f".{stem}.", suffix=".pt", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, artifact_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        manifest = {
            "schema_version": 1,
            "request_id": request_id,
            "sampling": sampling,
            "model_root": str(model_root),
            "component": component,
            "model": model,
            "capture": {
                "step_index": step_index,
                "actual_timestep": actual_timestep,
                "cfg_branch_index": branch_index,
            },
            "input": input_summary,
            "input_sha256": input_sha256,
            "artifact": {
                "path": artifact_path.name,
                "sha256": _sha256_file(artifact_path),
            },
        }
        manifest["manifest_sha256"] = _sha256_json(manifest)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.output_dir,
            prefix=f".{stem}.",
            suffix=".json",
            delete=False,
        ) as temporary:
            json.dump(manifest, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_manifest = Path(temporary.name)
        try:
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        captured.add(capture_key)
        return manifest_path
