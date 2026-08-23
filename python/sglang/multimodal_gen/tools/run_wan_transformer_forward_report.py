# SPDX-License-Identifier: Apache-2.0

"""Load real Wan components and qualify one worker-captured direct forward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.runtime.qualification.wan_transformer_capture import (
    load_wan_transformer_input_capture,
)
from sglang.multimodal_gen.tools.compare_wan_transformer_forward import (
    run_wan_transformer_forward_qualification,
    validate_wan_transformer_forward_report,
    write_wan_transformer_forward_report,
)


def _configure_standalone_layerwise_offload(
    model: Any, *, component_name: str, server_args: Any
) -> None:
    """Mirror worker layerwise setup after loading one standalone component."""

    if not server_args.has_layerwise_offload_components():
        return
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        configure_layerwise_offload_modules,
    )

    configure_layerwise_offload_modules(
        {component_name: model},
        server_args,
        component_names=(
            None
            if server_args.component_residency is not None
            else server_args.layerwise_offload_components
        ),
        warn_missing=(
            server_args.component_residency is not None
            or server_args.is_arg_explicitly_set("layerwise_offload_components")
            or server_args.is_arg_explicitly_set("dit_layerwise_offload")
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--run-order",
        choices=("reference-first", "candidate-first"),
        required=True,
    )
    parser.add_argument("--model-id")
    parser.add_argument("--master-port", type=int, default=30005)
    parser.add_argument("--reference-scheduler-port", type=int, required=True)
    parser.add_argument("--candidate-scheduler-port", type=int, required=True)
    parser.add_argument("--strict-ports", action="store_true")
    parser.add_argument("--reference-attention-backend", default="fa")
    parser.add_argument("--candidate-attention-backend", default="wan_hybrid")
    parser.add_argument("--candidate-attention-backend-config")
    parser.add_argument(
        "--candidate-backend-expectation",
        choices=("exercised", "temporal_fallback"),
        default="exercised",
    )
    parser.add_argument("--reference-transformer-weights-path")
    parser.add_argument("--candidate-transformer-weights-path")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    return parser


def build_direct_port_provenance(
    *,
    master_port: int,
    reference_scheduler_port: int,
    candidate_scheduler_port: int,
    strict_ports: bool,
) -> dict[str, int | bool]:
    ports = (master_port, reference_scheduler_port, candidate_scheduler_port)
    if any(isinstance(port, bool) or not isinstance(port, int) for port in ports):
        raise ValueError("direct qualification ports must be integers")
    if any(port < 1 or port > 65535 for port in ports):
        raise ValueError("direct qualification ports must be valid TCP ports")
    if not strict_ports:
        raise ValueError("direct qualification requires strict ports")
    if len(set(ports)) != len(ports):
        raise ValueError(
            "direct qualification requires distinct master/reference/candidate ports"
        )
    return {
        "master_port": master_port,
        "reference_scheduler_port": reference_scheduler_port,
        "candidate_scheduler_port": candidate_scheduler_port,
        "reference_strict_ports": True,
        "candidate_strict_ports": True,
    }


def _initialize_single_gpu_runtime(master_port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
        }
    )
    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if get_context()._server_args is None:
        get_context().set_server_args(SrtServerArgs(model_path="dummy"))


def _load_component(
    *,
    model_root: str,
    component_name: str,
    component_path: str,
    model_id: str | None,
    attention_backend: str,
    attention_backend_config: str | None,
    transformer_weights_path: str | None,
    scheduler_port: int,
    strict_ports: bool,
) -> Any:
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        PipelineComponentLoader,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    kwargs: dict[str, Any] = {
        "model_path": model_root,
        "backend": "sglang",
        "num_gpus": 1,
        "warmup_mode": "off",
        "scheduler_port": scheduler_port,
        "strict_ports": strict_ports,
        "attention_backend": attention_backend,
        "component_paths": {component_name: component_path},
        "component_attention_backends": {
            component_name: attention_backend,
        },
    }
    if model_id is not None:
        kwargs["model_id"] = model_id
    if attention_backend_config is not None:
        # Parse before ServerArgs so malformed JSON cannot affect model loading.
        kwargs["attention_backend_config"] = json.dumps(
            json.loads(attention_backend_config),
            sort_keys=True,
            separators=(",", ":"),
        )
    if transformer_weights_path is not None:
        if component_name == "transformer":
            kwargs["transformer_weights_path"] = transformer_weights_path
        else:
            kwargs["component_transformer_weights_paths"] = {
                component_name: transformer_weights_path
            }
    server_args = ServerArgs.from_kwargs(**kwargs)
    set_global_server_args(server_args)
    config = json.loads(
        (Path(component_path) / "config.json").read_text(encoding="utf-8")
    )
    architecture = config.get("_class_name")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("captured component config has no _class_name")
    backend, matched_name = server_args.resolve_component_attention_backend(
        component_name
    )
    if backend is None:
        raise RuntimeError("direct Wan qualification did not resolve attention backend")
    model, _memory_usage = PipelineComponentLoader.load_component(
        component_name=component_name,
        component_model_path=component_path,
        transformers_or_diffusers="diffusers",
        server_args=server_args,
        component_architecture=architecture,
        component_attn_backend=backend,
        component_attn_name=matched_name or component_name,
    )
    _configure_standalone_layerwise_offload(
        model,
        component_name=component_name,
        server_args=server_args,
    )
    model.eval()
    model._wan_qualification_server_args = server_args
    return model


def main() -> None:
    args = _build_parser().parse_args()
    port_provenance = build_direct_port_provenance(
        master_port=args.master_port,
        reference_scheduler_port=args.reference_scheduler_port,
        candidate_scheduler_port=args.candidate_scheduler_port,
        strict_ports=args.strict_ports,
    )
    capture_manifest_path = Path(args.capture_manifest).expanduser().resolve()

    # This validates manifest, request/sampling/model/component bindings, every
    # recorded component file, the tensor artifact, and all input digests before
    # distributed initialization or any model allocation begins.
    _, capture_manifest = load_wan_transformer_input_capture(capture_manifest_path)
    model_root = capture_manifest["model_root"]
    component_name = capture_manifest["component"]["name"]
    component_path = capture_manifest["component"]["resolved_path"]
    if component_name not in ("transformer", "transformer_2"):
        raise ValueError("capture did not identify a Wan transformer component")
    expected_component_path = (Path(model_root) / component_name).resolve()
    if Path(component_path).resolve() != expected_component_path:
        raise ValueError(
            "direct qualification requires the captured model component path"
        )

    _initialize_single_gpu_runtime(args.master_port)
    reference_model = _load_component(
        model_root=model_root,
        component_name=component_name,
        component_path=component_path,
        model_id=args.model_id,
        attention_backend=args.reference_attention_backend,
        attention_backend_config=None,
        transformer_weights_path=args.reference_transformer_weights_path,
        scheduler_port=args.reference_scheduler_port,
        strict_ports=args.strict_ports,
    )
    candidate_model = _load_component(
        model_root=model_root,
        component_name=component_name,
        component_path=component_path,
        model_id=args.model_id,
        attention_backend=args.candidate_attention_backend,
        attention_backend_config=args.candidate_attention_backend_config,
        transformer_weights_path=args.candidate_transformer_weights_path,
        scheduler_port=args.candidate_scheduler_port,
        strict_ports=args.strict_ports,
    )
    report = run_wan_transformer_forward_qualification(
        reference_model=reference_model,
        candidate_model=candidate_model,
        capture_manifest_path=capture_manifest_path,
        run_order=args.run_order,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        candidate_backend_expectation=args.candidate_backend_expectation,
    )
    report["port_provenance"] = port_provenance
    errors = validate_wan_transformer_forward_report(
        report,
        expected_warmup_runs=args.warmup_runs,
        expected_measure_runs=args.measure_runs,
        expected_model_path=model_root,
    )
    if errors:
        report["validation_errors"] = errors
    output_path = Path(args.output_json).expanduser().resolve()
    write_wan_transformer_forward_report(report, output_path)
    if errors or not report["qualification"]["passed"]:
        raise RuntimeError(
            f"Wan direct transformer qualification failed; see {output_path}"
        )


if __name__ == "__main__":
    main()
