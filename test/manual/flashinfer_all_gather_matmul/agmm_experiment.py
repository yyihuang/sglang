"""Experimental TP4 Llama-70B Q-projection all-gather matmul route."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

_LOG = logging.getLogger("sglang.agmm_experiment")
_VARIANT = os.environ.get("AGMM_EXPERIMENT_VARIANT", "")
_MIN_FULL_TOKENS = int(os.environ.get("AGMM_EXPERIMENT_MIN_FULL_TOKENS", "512"))
_VALIDATE_CALLS = int(os.environ.get("AGMM_EXPERIMENT_VALIDATE_CALLS", "80"))
_ARTIFACT_DIR = Path(os.environ.get("AGMM_EXPERIMENT_ARTIFACT_DIR", "/tmp"))
_LOCAL_INPUTS: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
_ENABLED_GROUPS: set[str] = set()
_COUNTERS = {
    "candidate_hits": 0,
    "cake_backend_requests": 0,
    "explicit_hits": 0,
    "native_small_m": 0,
    "full_tokens": 0,
    "padded_tokens": 0,
    "validated_calls": 0,
}
_SEEN_MODULES: set[int] = set()


def _tp_state():
    from sglang.srt.distributed import get_tp_group

    coordinator = get_tp_group()
    return coordinator, coordinator.device_group


def _summary_path(rank: int) -> Path:
    return _ARTIFACT_DIR / f"agmm-route-rank{rank}.json"


def _write_summary() -> None:
    if not dist.is_initialized():
        return
    coordinator, _ = _tp_state()
    payload = {
        "variant": _VARIANT,
        "candidate_backend": "cake" if _VARIANT == "candidate" else None,
        "rank": coordinator.rank_in_group,
        "world_size": coordinator.world_size,
        "min_full_tokens": _MIN_FULL_TOKENS,
        "unique_qkv_modules": len(_SEEN_MODULES),
        **_COUNTERS,
    }
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _summary_path(coordinator.rank_in_group).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


atexit.register(_write_summary)


def _is_llama70_tp4_qkv(layer, x: torch.Tensor) -> bool:
    return (
        type(layer).__name__ == "QKVParallelLinear"
        and getattr(layer, "tp_size", None) == 4
        and getattr(layer, "q_proj_shard_size", None) == 2048
        and getattr(layer, "kv_proj_shard_size", None) == 256
        and getattr(layer, "v_proj_shard_size", None) == 256
        and tuple(layer.weight.shape) == (2560, 8192)
        and layer.bias is None
        and x.ndim == 2
        and x.shape[1] == 8192
        and x.dtype == torch.bfloat16
        and x.is_cuda
    )


def _enable_symmetric_memory(group) -> None:
    if group.group_name in _ENABLED_GROUPS:
        return
    symm_mem.set_backend("NVSHMEM")
    symm_mem.enable_symm_mem_for_group(group.group_name)
    _ENABLED_GROUPS.add(group.group_name)


def _local_symmetric_input(
    x: torch.Tensor, *, rank: int, world_size: int
) -> tuple[torch.Tensor, int]:
    # The fused contract requires local M to be a positive multiple of 128.
    local_m = max(128, math.ceil(x.shape[0] / (128 * world_size)) * 128)
    full_padded_m = local_m * world_size
    key = (x.device.index, local_m, x.dtype)
    local = _LOCAL_INPUTS.get(key)
    if local is None:
        local = symm_mem.empty(local_m, x.shape[1], device=x.device, dtype=x.dtype)
        _LOCAL_INPUTS[key] = local

    start = rank * local_m
    stop = min(start + local_m, x.shape[0])
    if stop - start != local_m:
        local.zero_()
    if stop > start:
        local[: stop - start].copy_(x[start:stop])
    return local, full_padded_m


def _q_weight_kn(layer) -> torch.Tensor:
    cached = getattr(layer, "_agmm_q_weight_kn", None)
    if cached is None:
        cached = layer.weight[: layer.q_proj_shard_size].t().contiguous()
        layer._agmm_q_weight_kn = cached
    return cached


def _validate(layer, x: torch.Tensor, result: torch.Tensor) -> None:
    if _COUNTERS["validated_calls"] >= _VALIDATE_CALLS:
        return
    expected = F.linear(x, layer.weight, None)
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    _COUNTERS["validated_calls"] += 1


def maybe_qkv_forward(layer, input_: torch.Tensor):
    """Return an experimental forward result, or ``None`` for native SGLang."""
    if _VARIANT not in ("explicit", "candidate"):
        return None
    if not _is_llama70_tp4_qkv(layer, input_):
        return None
    if input_.shape[0] < _MIN_FULL_TOKENS:
        _COUNTERS["native_small_m"] += 1
        return None

    coordinator, group = _tp_state()
    if coordinator.world_size != 4:
        raise RuntimeError(
            f"AGMM experiment requires TP world size 4, got {coordinator.world_size}"
        )
    if torch.cuda.get_device_capability(input_.device)[0] != 10:
        raise RuntimeError("AGMM experiment requires an SM100/SM103 GPU")

    _enable_symmetric_memory(group)
    local, full_padded_m = _local_symmetric_input(
        input_, rank=coordinator.rank_in_group, world_size=coordinator.world_size
    )
    q_weight_kn = _q_weight_kn(layer)

    if _VARIANT == "explicit":
        gathered = input_.new_empty(full_padded_m, input_.shape[1])
        dist.all_gather_into_tensor(gathered, local, group=group)
        q = gathered[: input_.shape[0]] @ q_weight_kn
        _COUNTERS["explicit_hits"] += 1
    else:
        from flashinfer.comm import all_gather_matmul

        q = all_gather_matmul(local, q_weight_kn, group, backend="cake")
        q = q[: input_.shape[0]]
        _COUNTERS["candidate_hits"] += 1
        _COUNTERS["cake_backend_requests"] += 1

    kv = F.linear(input_, layer.weight[layer.q_proj_shard_size :], None)
    result = torch.cat((q, kv), dim=-1)
    _validate(layer, input_, result)

    _SEEN_MODULES.add(id(layer))
    _COUNTERS["full_tokens"] += input_.shape[0]
    _COUNTERS["padded_tokens"] += full_padded_m
    hits = _COUNTERS[f"{_VARIANT}_hits"]
    if hits == 1 or hits % 80 == 0:
        _write_summary()
        if coordinator.rank_in_group == 0:
            _LOG.info(
                "AGMM_ROUTE variant=%s backend=%s hits=%d modules=%d full_m=%d padded_m=%d",
                _VARIANT,
                "cake" if _VARIANT == "candidate" else "explicit",
                hits,
                len(_SEEN_MODULES),
                input_.shape[0],
                full_padded_m,
            )
    return result, None
