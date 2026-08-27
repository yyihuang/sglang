# SPDX-License-Identifier: Apache-2.0

"""Runtime-owned attention backend identity for Wan qualification requests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _class_identity(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def collect_runtime_attention_backend_identity(
    transformers: Iterable[Any], *, requested_backend: str | None
) -> dict[str, Any] | None:
    """Summarize resolver and execution evidence from live attention layers.

    The resolver class is attached by the layer constructor from the class that
    ``get_attn_backend`` actually returned.  The FlashAttention implementation
    records its version only after a real attention call returns successfully.
    This helper therefore does not infer either value from the GPU architecture
    or from a backend name.
    """

    if requested_backend is None or requested_backend.lower() != "fa":
        return None

    expected_instance_count = 0
    observations: list[tuple[str, str, int]] = []
    for transformer in transformers:
        if transformer is None:
            continue
        named_modules = getattr(transformer, "named_modules", None)
        modules = (
            named_modules()
            if callable(named_modules)
            else (("", module) for module in transformer.modules())
        )
        for module_name, module in modules:
            # A candidate-configured Wan model retains a dormant FA fallback
            # beside each self-attention layer. A reference request switches the
            # active layer itself to FA, so the fallback is intentionally not
            # executed and must not count as an expected runtime observation.
            if module_name.endswith("attn1_fallback"):
                continue
            backend = getattr(module, "backend", None)
            if getattr(backend, "name", None) != "FA":
                continue
            expected_instance_count += 1
            backend_cls = getattr(module, "_resolved_attn_backend_cls", None)
            impl = getattr(module, "attn_impl", None)
            version = getattr(impl, "_runtime_observed_flash_attention_version", None)
            if not isinstance(backend_cls, type) or not isinstance(version, int):
                continue
            observations.append(
                (_class_identity(backend_cls), _class_identity(type(impl)), version)
            )

    unique_observations = sorted(set(observations))
    fully_observed = (
        expected_instance_count > 0
        and len(observations) == expected_instance_count
        and len(unique_observations) == 1
    )
    resolved_backend_class = None
    resolved_impl_class = None
    flash_attention_version = None
    if len(unique_observations) == 1:
        (
            resolved_backend_class,
            resolved_impl_class,
            flash_attention_version,
        ) = unique_observations[0]

    return {
        "requested_backend": "fa",
        "resolved_backend_class": resolved_backend_class,
        "resolved_impl_class": resolved_impl_class,
        "implementation": (
            f"FA{flash_attention_version}"
            if fully_observed and flash_attention_version is not None
            else None
        ),
        "flash_attention_version": flash_attention_version,
        "runtime_observed": fully_observed,
        "expected_instance_count": expected_instance_count,
        "observed_instance_count": len(observations),
    }
