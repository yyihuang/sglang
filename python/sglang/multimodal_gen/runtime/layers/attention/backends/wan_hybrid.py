# SPDX-License-Identifier: Apache-2.0

import threading
import weakref
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class _WanHybridSharedScratch:
    workspace: object


_SHARED_SCRATCH_LOCK = threading.Lock()
_SHARED_SCRATCH: weakref.WeakValueDictionary[tuple, _WanHybridSharedScratch] = (
    weakref.WeakValueDictionary()
)


@dataclass
class WanHybridEvidenceCollector:
    """Evidence owned by exactly one Wan serving request."""

    request_id: str | None = None
    route_events: list[dict[str, Any]] = field(default_factory=list)
    success_events: list[dict[str, Any]] = field(default_factory=list)
    raw_success_count: int = 0

    def record_route(self, event: dict[str, Any]) -> None:
        self.route_events.append(dict(event))

    def record_success(self, event: dict[str, Any] | None) -> None:
        self.raw_success_count += 1
        if event is not None:
            self.success_events.append(dict(event))

    def hit_count(self) -> int:
        return self.raw_success_count

    def coverage(self) -> dict[str, Any]:
        return _build_wan_hybrid_coverage(
            self.route_events,
            self.success_events,
            self.raw_success_count,
            request_id=self.request_id,
        )


_STANDALONE_EVIDENCE: ContextVar[WanHybridEvidenceCollector] = ContextVar(
    "wan_hybrid_standalone_evidence", default=WanHybridEvidenceCollector()
)


def reset_wan_hybrid_hit_count() -> None:
    """Reset context-local standalone evidence used outside serving requests."""

    _STANDALONE_EVIDENCE.set(WanHybridEvidenceCollector())


def read_wan_hybrid_hit_count() -> int:
    return _STANDALONE_EVIDENCE.get().hit_count()


def _current_evidence_coordinates(
    layer_index: int,
) -> tuple[WanHybridEvidenceCollector, dict[str, Any]] | None:
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context,
    )

    try:
        context = get_forward_context()
    except AssertionError:
        return None
    collector = context.wan_hybrid_evidence_collector
    if collector is None:
        return None
    coordinates = {
        "step_index": context.current_timestep,
        "actual_timestep": context.wan_actual_timestep,
        "component_name": context.wan_component_name,
        "cfg_branch_index": context.wan_cfg_branch_index,
        "layer_index": layer_index,
    }
    missing = [name for name, value in coordinates.items() if value is None]
    if missing:
        raise RuntimeError(
            "Wan hybrid evidence context is incomplete: " + ", ".join(missing)
        )
    return collector, coordinates


def record_wan_attention_route(
    *,
    layer_index: int,
    hybrid_configured: bool,
    eligible_for_hybrid: bool,
) -> None:
    """Record the actual self-attention route before a Wan block executes it."""

    evidence = _current_evidence_coordinates(layer_index)
    if evidence is None:
        return
    collector, coordinates = evidence
    if eligible_for_hybrid and not hybrid_configured:
        raise RuntimeError("Wan hybrid evidence marked an FA-only layer as eligible")
    event = coordinates | {
        "hybrid_configured": hybrid_configured,
        "eligible_for_hybrid": eligible_for_hybrid,
        "planned_backend": "wan_hybrid" if eligible_for_hybrid else "fa",
        "configured_fallback": hybrid_configured and not eligible_for_hybrid,
        "control": not hybrid_configured,
    }
    collector.record_route(event)


def _record_successful_wan_hybrid_forward(
    result: torch.Tensor, *, layer_index: int | None
) -> torch.Tensor:
    if layer_index is None:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        try:
            context = get_forward_context()
        except AssertionError:
            context = None
        if (
            context is not None
            and context.wan_hybrid_evidence_collector is not None
        ):
            raise RuntimeError(
                "Wan hybrid evidence cannot attribute a successful call without "
                "a layer index"
            )
    evidence = (
        _current_evidence_coordinates(layer_index)
        if layer_index is not None
        else None
    )
    if evidence is None:
        _STANDALONE_EVIDENCE.get().record_success(None)
    else:
        collector, coordinates = evidence
        collector.record_success(coordinates)
    return result


def read_wan_hybrid_coverage() -> dict[str, Any]:
    """Return context-local standalone evidence outside serving requests."""

    return _STANDALONE_EVIDENCE.get().coverage()


def _build_wan_hybrid_coverage(
    route_events: list[dict[str, Any]],
    success_events: list[dict[str, Any]],
    actual_hit_count: int,
    *,
    request_id: str | None,
) -> dict[str, Any]:
    """Group planned routes and independently attributed successful calls."""

    routes = [dict(event) for event in route_events]
    successes = [dict(event) for event in success_events]

    success_counts = Counter(
        (
            event["step_index"],
            event["actual_timestep"],
            event["component_name"],
            event["cfg_branch_index"],
            event["layer_index"],
        )
        for event in successes
    )
    grouped: dict[tuple[int, int, str], dict[int, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for event in routes:
        grouped[
            (
                event["step_index"],
                event["actual_timestep"],
                event["component_name"],
            )
        ][event["cfg_branch_index"]].append(event)

    steps = []
    expected_hit_count = 0
    attributed_actual_hit_count = 0
    eligible_hybrid_miss_count = 0
    for (step_index, actual_timestep, component_name), branch_groups in sorted(
        grouped.items()
    ):
        branches = []
        for branch_index, branch_routes in sorted(branch_groups.items()):
            branch_routes.sort(key=lambda event: event["layer_index"])
            layer_indices = [event["layer_index"] for event in branch_routes]
            eligible_layers = [
                event["layer_index"]
                for event in branch_routes
                if event["eligible_for_hybrid"]
            ]
            planned_hybrid_layers = [
                event["layer_index"]
                for event in branch_routes
                if event["planned_backend"] == "wan_hybrid"
            ]
            configured_fallback_layers = [
                event["layer_index"]
                for event in branch_routes
                if event["configured_fallback"]
            ]
            control_layers = [
                event["layer_index"]
                for event in branch_routes
                if event["control"]
            ]
            successful_layers = []
            for layer_index in sorted(set(layer_indices)):
                successful_layers.extend(
                    [layer_index]
                    * success_counts[
                        (
                            step_index,
                            actual_timestep,
                            component_name,
                            branch_index,
                            layer_index,
                        )
                    ]
                )
            planned_counts = Counter(planned_hybrid_layers)
            successful_counts = Counter(successful_layers)
            eligible_misses = list((planned_counts - successful_counts).elements())
            unexpected_successes = list(
                (successful_counts - planned_counts).elements()
            )
            branch_expected = len(planned_hybrid_layers)
            branch_actual = len(successful_layers)
            expected_hit_count += branch_expected
            attributed_actual_hit_count += branch_actual
            eligible_hybrid_miss_count += len(eligible_misses)
            branches.append(
                {
                    "cfg_branch_index": branch_index,
                    "num_layers": len(layer_indices),
                    "layer_indices": layer_indices,
                    "eligible_layer_indices": eligible_layers,
                    "planned_hybrid_layer_indices": planned_hybrid_layers,
                    "successful_hybrid_layer_indices": successful_layers,
                    "eligible_hybrid_miss_layer_indices": eligible_misses,
                    "unexpected_successful_hybrid_layer_indices": unexpected_successes,
                    "configured_fallback_layer_indices": configured_fallback_layers,
                    "control_layer_indices": control_layers,
                    "expected_hit_count": branch_expected,
                    "actual_hit_count": branch_actual,
                }
            )
        steps.append(
            {
                "step_index": step_index,
                "actual_timestep": actual_timestep,
                "active_component": component_name,
                "executed_cfg_branch_indices": sorted(branch_groups),
                "branches": branches,
            }
        )

    return {
        "schema_version": 2,
        "request_id": request_id,
        "expected_hit_count": expected_hit_count,
        "actual_hit_count": actual_hit_count,
        "attributed_actual_hit_count": attributed_actual_hit_count,
        "unattributed_actual_hit_count": actual_hit_count
        - attributed_actual_hit_count,
        "eligible_hybrid_miss_count": eligible_hybrid_miss_count,
        "num_route_events": len(routes),
        "num_success_events": len(successes),
        "steps": steps,
    }


class WanHybridAttentionBackend(AttentionBackend):
    """Exact-shape Wan self-attention through FlashInfer's public API."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.WAN_HYBRID

    @staticmethod
    def get_impl_cls() -> type["WanHybridAttentionImpl"]:
        return WanHybridAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class WanHybridAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del extra_impl_args
        if head_size != 128:
            raise ValueError(
                f"Wan hybrid attention requires head_size=128, got {head_size}"
            )
        if causal:
            raise ValueError("Wan hybrid serving supports noncausal attention only")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads != num_heads:
            raise ValueError(
                "Wan hybrid attention requires equal query and KV head counts, "
                f"got {num_heads} and {num_kv_heads}"
            )
        if num_heads != 40:
            raise ValueError(
                f"Wan hybrid Wan serving requires num_heads=40, got {num_heads}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        prefix_parts = prefix.split(".")
        self._wan_layer_index = None
        for part_index in range(len(prefix_parts) - 2):
            if (
                prefix_parts[part_index] == "blocks"
                and prefix_parts[part_index + 1].isdigit()
                and prefix_parts[part_index + 2] == "attn1"
            ):
                self._wan_layer_index = int(prefix_parts[part_index + 1])
                break
        self._workspace_key = None
        self._shared_scratch = None
        self._output = None

    def _get_workspace_and_output(self, query: torch.Tensor):
        try:
            from flashinfer import WanHybridAttentionWorkspace
        except ImportError as error:
            raise ImportError(
                "Wan hybrid attention requires a FlashInfer build that exports "
                "WanHybridAttentionWorkspace."
            ) from error

        key = (
            tuple(query.shape),
            query.device.type,
            query.device.index,
            query.dtype,
            torch.cuda.current_stream(query.device).cuda_stream,
        )
        if key != self._workspace_key:
            with _SHARED_SCRATCH_LOCK:
                shared_scratch = _SHARED_SCRATCH.get(key)
                if shared_scratch is None:
                    shared_scratch = _WanHybridSharedScratch(
                        workspace=WanHybridAttentionWorkspace(query.device)
                    )
                    _SHARED_SCRATCH[key] = shared_scratch
            self._shared_scratch = shared_scratch
            self._output = torch.empty_like(query, dtype=torch.bfloat16)
            self._workspace_key = key
        return self._shared_scratch.workspace, self._output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        del attn_metadata
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "Wan hybrid attention supports dense self-attention only; "
                f"got q={tuple(query.shape)}, k={tuple(key.shape)}, "
                f"v={tuple(value.shape)}"
            )
        if query.ndim != 4 or query.shape[2:] != (
            self.num_heads,
            self.head_size,
        ):
            raise ValueError(
                "Wan hybrid attention expects [batch, seq_len, heads, 128], "
                f"got {tuple(query.shape)}"
            )
        if query.shape[0] != 1:
            raise ValueError(
                "Wan hybrid Wan serving is qualified only for batch=1, "
                f"got batch={query.shape[0]}"
            )
        if query.shape[1] != 4800:
            raise ValueError(
                "Wan hybrid Wan serving is qualified only for sequence length "
                f"4800, got {query.shape[1]}"
            )
        if query.device.type != "cuda":
            raise ValueError("Wan hybrid attention requires CUDA tensors")
        if query.dtype != torch.bfloat16:
            raise ValueError(
                "Wan hybrid serving requires BF16 Q/K/V because its output is BF16, "
                f"got {query.dtype}"
            )
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("Wan hybrid attention requires matching Q/K/V dtypes")
        if (
            not query.is_contiguous()
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError("Wan hybrid attention requires contiguous NHD Q/K/V")

        try:
            from flashinfer import (
                is_wan_hybrid_attention_available,
                wan_hybrid_attention,
            )
        except ImportError as error:
            raise ImportError(
                "Wan hybrid attention requires a FlashInfer build that exports "
                "the public wan_hybrid attention API."
            ) from error

        if not is_wan_hybrid_attention_available(query.device):
            raise NotImplementedError(
                "FlashInfer wan_hybrid attention is unavailable on this device"
            )

        workspace, output = self._get_workspace_and_output(query)
        return _record_successful_wan_hybrid_forward(
            wan_hybrid_attention(
                query,
                key,
                value,
                out=output,
                workspace=workspace,
                sm_scale=self.softmax_scale,
                qkv_layout="NHD",
                causal=False,
            ),
            layer_index=self._wan_layer_index,
        )
