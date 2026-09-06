# SPDX-License-Identifier: Apache-2.0

"""Capture exact real Wan transformer call kwargs from one serving request."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator

from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    CAPTURE_BRANCH_ENV,
    CAPTURE_COMPONENTS_ENV,
    CAPTURE_DIR_ENV,
    CAPTURE_REQUEST_ID_ENV,
    CAPTURE_STEPS_ENV,
    WAN_COMPONENT_NAMES,
    load_wan_transformer_input_capture,
)


def _parse_assignments(values: list[str], *, option: str) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, selected = value.partition("=")
        if separator != "=" or not name or not selected:
            raise ValueError(f"{option} entries must use name=value")
        if name in result:
            raise ValueError(f"{option} contains duplicate name {name!r}")
        result[name] = selected
    return result


@contextlib.contextmanager
def _capture_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-index-json", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master-port", type=int, default=30005)
    parser.add_argument("--attention-backend", default="fa")
    parser.add_argument("--attention-backend-config")
    parser.add_argument("--transformer-weights-path")
    parser.add_argument(
        "--component",
        action="append",
        choices=WAN_COMPONENT_NAMES,
        dest="components",
    )
    parser.add_argument(
        "--component-step",
        action="append",
        default=[],
        help="Optional component=step selection; omitted means first active call.",
    )
    parser.add_argument(
        "--cfg-branch-index",
        action="append",
        type=int,
        dest="cfg_branch_indices",
        help="CFG branch to capture; repeat to capture multiple branches.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Capture every executed step for the selected components/branches.",
    )
    parser.add_argument("--component-path", action="append", default=[])
    parser.add_argument("--component-attention-backend", action="append", default=[])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    selected_components = args.components or list(WAN_COMPONENT_NAMES)
    if len(selected_components) != len(set(selected_components)):
        raise ValueError("--component contains a duplicate component")
    components = tuple(selected_components)
    component_steps = _parse_assignments(args.component_step, option="--component-step")
    if set(component_steps) - set(components):
        raise ValueError("--component-step selected a component not being captured")
    for component, step in component_steps.items():
        if component not in WAN_COMPONENT_NAMES or int(step) < 0:
            raise ValueError(
                "--component-step must use a Wan component and nonnegative step"
            )
    cfg_branch_indices = args.cfg_branch_indices or [0]
    if any(branch < 0 for branch in cfg_branch_indices):
        raise ValueError("--cfg-branch-index must be nonnegative")
    if len(cfg_branch_indices) != len(set(cfg_branch_indices)):
        raise ValueError("--cfg-branch-index contains a duplicate branch")
    if args.all_steps and component_steps:
        raise ValueError("--all-steps cannot be combined with --component-step")

    request_id = f"wan-transformer-capture-{uuid.uuid4()}"
    request_dir = Path(args.output_dir).expanduser().resolve() / request_id
    request_dir.mkdir(parents=True, exist_ok=False)
    output_index = Path(args.output_index_json).expanduser().resolve()
    output_index.parent.mkdir(parents=True, exist_ok=True)

    server_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "backend": "sglang",
        "num_gpus": 1,
        "master_port": args.master_port,
        "warmup_mode": "off",
        "attention_backend": args.attention_backend,
    }
    if args.model_id:
        server_kwargs["model_id"] = args.model_id
    if args.attention_backend_config:
        server_kwargs["attention_backend_config"] = args.attention_backend_config
    if args.transformer_weights_path:
        server_kwargs["transformer_weights_path"] = args.transformer_weights_path
    component_paths = _parse_assignments(args.component_path, option="--component-path")
    if component_paths:
        server_kwargs["component_paths"] = component_paths
    component_backends = _parse_assignments(
        args.component_attention_backend,
        option="--component-attention-backend",
    )
    if component_backends:
        server_kwargs["component_attention_backends"] = component_backends

    sampling_kwargs = {
        "request_id": request_id,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "seed": args.seed,
        "return_frames": True,
        "save_output": False,
    }
    capture_env = {
        CAPTURE_DIR_ENV: str(request_dir),
        CAPTURE_REQUEST_ID_ENV: request_id,
        CAPTURE_COMPONENTS_ENV: ",".join(components),
        CAPTURE_BRANCH_ENV: ",".join(str(index) for index in cfg_branch_indices),
        CAPTURE_STEPS_ENV: (
            "*"
            if args.all_steps
            else ",".join(
                f"{component}={step}" for component, step in component_steps.items()
            )
        ),
    }
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    with _capture_environment(capture_env):
        with DiffGenerator.from_pretrained(
            local_mode=True, **server_kwargs
        ) as generator:
            result = generator.generate(sampling_params_kwargs=sampling_kwargs)
    if result is None or isinstance(result, list):
        raise RuntimeError("singleton Wan capture request did not return one result")
    result_request_id = getattr(result, "metrics", {}).get("request_id")
    if result_request_id != request_id:
        raise RuntimeError("Wan capture result request_id does not match request")

    manifests = sorted(request_dir.glob("*.manifest.json"))
    if not args.all_steps and len(manifests) != len(components):
        raise RuntimeError(
            "Wan capture must produce exactly one artifact per requested component; "
            f"expected {len(components)}, found {len(manifests)}"
        )
    records = []
    observed_components = []
    for manifest_path in manifests:
        _, manifest = load_wan_transformer_input_capture(manifest_path)
        if manifest["request_id"] != request_id:
            raise RuntimeError("Wan capture artifact request_id does not match request")
        observed_components.append(manifest["component"]["name"])
        records.append(
            {
                "component_name": manifest["component"]["name"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "artifact_sha256": manifest["artifact"]["sha256"],
                "capture": dict(manifest["capture"]),
            }
        )
    if set(observed_components) != set(components):
        raise RuntimeError(
            "Wan capture components are missing or duplicated: "
            f"{observed_components}"
        )
    coordinate_inventory = [
        {
            "component_name": record["component_name"],
            **record["capture"],
        }
        for record in records
    ]
    if args.all_steps:
        expected_coordinates = {
            (step_index, branch_index)
            for step_index in range(args.num_inference_steps)
            for branch_index in cfg_branch_indices
        }
        actual_coordinates = {
            (
                record["capture"]["step_index"],
                record["capture"]["cfg_branch_index"],
            )
            for record in records
        }
        if actual_coordinates != expected_coordinates or len(records) != len(
            expected_coordinates
        ):
            raise RuntimeError(
                "Wan all-step capture did not produce exactly one active component "
                "artifact per requested step/branch"
            )
    index = {
        "schema_version": 2,
        "request_id": request_id,
        "capture_mode": "all_steps" if args.all_steps else "selected_steps",
        "requested_components": list(components),
        "requested_cfg_branch_indices": cfg_branch_indices,
        "num_inference_steps": args.num_inference_steps,
        "coordinate_inventory": coordinate_inventory,
        "components": records,
    }
    output_index.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
