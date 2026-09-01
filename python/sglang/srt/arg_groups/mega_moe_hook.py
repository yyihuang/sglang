from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    validate_flashinfer_megamoe_args(server_args)


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "megamoe":
        return

    if cfg.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            cfg.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )


def validate_flashinfer_megamoe_args(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "flashinfer_megamoe":
        return

    if cfg.flashinfer_megamoe_max_tokens_per_rank != 128:
        raise ValueError(
            "--flashinfer-megamoe-max-tokens-per-rank is fixed at 128 for the "
            "SM100 BF16 rank-major backend."
        )
    if cfg.moe_runner_backend != "auto":
        raise ValueError(
            "--moe-a2a-backend flashinfer_megamoe owns dispatch, expert "
            "compute, and combine; use --moe-runner-backend auto."
        )
    if cfg.enable_w4a4_mxfp4_megamoe:
        raise ValueError(
            "--enable-w4a4-mxfp4-megamoe is a DeepGEMM MegaMoE option and "
            "cannot be used with --moe-a2a-backend flashinfer_megamoe."
        )
    if cfg.enable_waterfill or cfg.enable_eplb or cfg.ep_num_redundant_experts:
        raise ValueError(
            "--moe-a2a-backend flashinfer_megamoe currently requires a static "
            "expert placement without Waterfill, EPLB, or redundant experts."
        )
