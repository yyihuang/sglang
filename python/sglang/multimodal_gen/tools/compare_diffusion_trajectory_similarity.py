"""Compare diffusion BF16 and quantized runs via trajectory-latent similarity.

This tool runs two SGLang diffusion variants with the same prompt and seed,
captures intermediate denoising latents via `return_trajectory_latents`, and
reports cosine / error metrics for each timestep plus final frame metrics.

The intended use is quant validation with reduced deterministic settings:
- same prompt / seed / resolution / step count for both variants
- BF16 reference on the base model
- FP8 candidate via `--candidate-transformer-path` and/or component overrides

Example:

    python -m sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity \
        --model-path /path/to/model \
        --prompt "A futuristic cyberpunk city at night" \
        --width 512 --height 512 --num-inference-steps 8 --seed 42 \
        --text-encoder-cpu-offload \
        --candidate-transformer-path /tmp/modelopt_flux2_fp8/sglang_transformer \
        --output-json /tmp/flux2_similarity.json
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


MODEL_QUALIFICATION_THRESHOLDS = {
    "atol": 1.0,
    "rtol": 0.1,
    "cosine_min": 0.995,
    "mae_max": 0.025,
    "repeatability_atol": 0.0,
    "repeatability_rtol": 0.0,
    "speedup_min": 1.0,
}


def parse_component_overrides(entries: Sequence[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected format component=path."
            )
        component, path = entry.split("=", 1)
        component = component.strip().replace("-", "_")
        path = path.strip()
        if not component or not path:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected format component=path."
            )
        overrides[component] = path
    return overrides


def parse_json_object(value: str | None, *, option_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {option_name} JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def _cosine_similarity(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    norm_a = torch.linalg.vector_norm(flat_a).item()
    norm_b = torch.linalg.vector_norm(flat_b).item()
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = float(F.cosine_similarity(flat_a, flat_b, dim=0).item())
    return max(-1.0, min(1.0, similarity))


def compute_tensor_metrics(lhs: Any, rhs: Any) -> dict[str, Any]:
    lhs_tensor = torch.as_tensor(lhs).detach().cpu().float()
    rhs_tensor = torch.as_tensor(rhs).detach().cpu().float()
    if lhs_tensor.shape != rhs_tensor.shape:
        raise ValueError(
            f"Metric shape mismatch: {tuple(lhs_tensor.shape)} vs {tuple(rhs_tensor.shape)}"
        )

    diff = lhs_tensor - rhs_tensor
    mse = float(diff.square().mean().item())
    rmse = float(math.sqrt(mse))
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    cosine = _cosine_similarity(lhs_tensor.reshape(-1), rhs_tensor.reshape(-1))
    return {
        "cosine_similarity": cosine,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
        "l2": l2,
        "finite": bool(
            torch.isfinite(lhs_tensor).all().item()
            and torch.isfinite(rhs_tensor).all().item()
        ),
        "within_tolerance": bool(
            torch.allclose(
                rhs_tensor,
                lhs_tensor,
                atol=MODEL_QUALIFICATION_THRESHOLDS["atol"],
                rtol=MODEL_QUALIFICATION_THRESHOLDS["rtol"],
            )
        ),
        "exact_match": bool(torch.equal(lhs_tensor, rhs_tensor)),
    }


def compute_uint8_frame_metrics(lhs: Any, rhs: Any) -> dict[str, float]:
    metrics = compute_tensor_metrics(lhs, rhs)
    mse = metrics["mse"]
    metrics["psnr_db"] = (
        float("inf") if mse == 0.0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
    )
    return metrics


def _normalize_step_index(step_index: int, num_steps: int) -> int:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if step_index < 0:
        step_index += num_steps
    if step_index < 0 or step_index >= num_steps:
        raise IndexError(
            f"Requested step index {step_index} is outside the valid range [0, {num_steps})."
        )
    return step_index


def _maybe_scalar(timestep: torch.Tensor | None, index: int) -> float | None:
    if timestep is None:
        return None
    value = timestep[index]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
    return float(value)


def summarize_trajectory_metrics(
    reference_latents: Any,
    candidate_latents: Any,
    *,
    reference_timesteps: Any = None,
    candidate_timesteps: Any = None,
    step_index: int = -1,
) -> dict[str, Any]:
    ref = torch.as_tensor(reference_latents).detach().cpu().float()
    cand = torch.as_tensor(candidate_latents).detach().cpu().float()
    if ref.shape != cand.shape:
        raise ValueError(
            f"Trajectory shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}"
        )
    if ref.ndim < 2:
        raise ValueError(
            f"Expected trajectory latents with an explicit timestep dimension, got {tuple(ref.shape)}"
        )

    num_steps = ref.shape[1]
    selected_step = _normalize_step_index(step_index, num_steps)
    ref_t = (
        torch.as_tensor(reference_timesteps).detach().cpu()
        if reference_timesteps is not None
        else None
    )
    cand_t = (
        torch.as_tensor(candidate_timesteps).detach().cpu()
        if candidate_timesteps is not None
        else None
    )
    timesteps_available = ref_t is not None and cand_t is not None
    timesteps_finite = bool(
        timesteps_available
        and torch.isfinite(ref_t).all().item()
        and torch.isfinite(cand_t).all().item()
    )
    timesteps_match = bool(
        timesteps_available
        and ref_t.shape == cand_t.shape
        and torch.equal(ref_t, cand_t)
    )

    per_step: list[dict[str, Any]] = []
    for idx in range(num_steps):
        metrics = compute_tensor_metrics(ref[:, idx], cand[:, idx])
        metrics["step_index"] = idx
        metrics["reference_timestep"] = _maybe_scalar(ref_t, idx)
        metrics["candidate_timestep"] = _maybe_scalar(cand_t, idx)
        per_step.append(metrics)

    return {
        "trajectory_shape": list(ref.shape),
        "num_steps": num_steps,
        "selected_step_index": selected_step,
        "timesteps_available": timesteps_available,
        "timesteps_finite": timesteps_finite,
        "timesteps_match": timesteps_match,
        "selected_step_metrics": per_step[selected_step],
        "per_step_metrics": per_step,
    }


def summarize_output_frame_metrics(
    reference_frames: Sequence[Any],
    candidate_frames: Sequence[Any],
) -> dict[str, Any]:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError(
            f"Output frame count mismatch: {len(reference_frames)} vs {len(candidate_frames)}"
        )
    if not reference_frames:
        raise ValueError("No output frames available for comparison.")

    ref_stack = np.stack([np.asarray(frame) for frame in reference_frames], axis=0)
    cand_stack = np.stack([np.asarray(frame) for frame in candidate_frames], axis=0)

    frame0_metrics = compute_uint8_frame_metrics(ref_stack[0], cand_stack[0])
    mid_index = len(reference_frames) // 2
    mid_metrics = compute_uint8_frame_metrics(
        ref_stack[mid_index], cand_stack[mid_index]
    )
    all_metrics = compute_uint8_frame_metrics(ref_stack, cand_stack)

    return {
        "num_frames": len(reference_frames),
        "frame0_metrics": frame0_metrics,
        "mid_frame_index": mid_index,
        "mid_frame_metrics": mid_metrics,
        "all_frames_metrics": all_metrics,
    }


def _summarize_result_pair(
    reference: Any,
    candidate: Any,
    *,
    reference_run_index: int,
    candidate_run_index: int,
    step_index: int,
) -> dict[str, Any]:
    return {
        "reference_run_index": reference_run_index,
        "candidate_run_index": candidate_run_index,
        "trajectory_metrics": summarize_trajectory_metrics(
            reference.trajectory_latents,
            candidate.trajectory_latents,
            reference_timesteps=reference.trajectory_timesteps,
            candidate_timesteps=candidate.trajectory_timesteps,
            step_index=step_index,
        ),
        "output_metrics": summarize_output_frame_metrics(
            extract_result_frames(reference),
            extract_result_frames(candidate),
        ),
    }


def _summarize_comparison_envelope(
    comparisons: Sequence[dict[str, Any]],
) -> dict[str, float]:
    if not comparisons:
        raise ValueError("At least one result comparison is required.")

    selected_trajectory = [
        comparison["trajectory_metrics"]["selected_step_metrics"]
        for comparison in comparisons
    ]
    all_steps_trajectory = [
        metrics
        for comparison in comparisons
        for metrics in comparison["trajectory_metrics"]["per_step_metrics"]
    ]
    all_frames = [
        comparison["output_metrics"]["all_frames_metrics"]
        for comparison in comparisons
    ]
    return {
        "min_selected_trajectory_cosine": min(
            metrics["cosine_similarity"] for metrics in selected_trajectory
        ),
        "max_selected_trajectory_mae": max(
            metrics["mae"] for metrics in selected_trajectory
        ),
        "min_all_steps_trajectory_cosine": min(
            metrics["cosine_similarity"] for metrics in all_steps_trajectory
        ),
        "max_all_steps_trajectory_mae": max(
            metrics["mae"] for metrics in all_steps_trajectory
        ),
        "max_all_steps_trajectory_max_abs": max(
            metrics["max_abs"] for metrics in all_steps_trajectory
        ),
        "min_all_frames_cosine": min(
            metrics["cosine_similarity"] for metrics in all_frames
        ),
        "max_all_frames_mae": max(metrics["mae"] for metrics in all_frames),
        "min_all_frames_psnr_db": min(metrics["psnr_db"] for metrics in all_frames),
    }


def summarize_run_repeatability(
    measured_results: Sequence[Any], *, step_index: int = -1
) -> dict[str, Any]:
    """Measure the worst same-variant delta across every measured-run pair.

    Diffusion backends can be nondeterministic even with an identical seed.  The
    resulting envelope is needed before attributing a reference/candidate delta
    to a quantized attention backend.
    """
    num_runs = len(measured_results)
    if num_runs < 2:
        return {
            "available": False,
            "num_runs": num_runs,
            "reason": "repeatability requires at least two measured runs",
        }

    comparisons = [
        _summarize_result_pair(
            measured_results[reference_run_index],
            measured_results[candidate_run_index],
            reference_run_index=reference_run_index,
            candidate_run_index=candidate_run_index,
            step_index=step_index,
        )
        for reference_run_index, candidate_run_index in itertools.combinations(
            range(num_runs), 2
        )
    ]
    return {
        "available": True,
        "num_runs": num_runs,
        "pairing": "all-pairs",
        "num_pairs": len(comparisons),
        "comparisons": comparisons,
        "envelope": _summarize_comparison_envelope(comparisons),
    }


def summarize_cross_variant_metrics(
    reference_results: Sequence[Any],
    candidate_results: Sequence[Any],
    *,
    step_index: int = -1,
) -> dict[str, Any]:
    """Measure every reference/candidate pair, not only each variant's last run."""
    if not reference_results or not candidate_results:
        raise ValueError("Cross-variant metrics require both measured result sets.")

    comparisons = [
        _summarize_result_pair(
            reference,
            candidate,
            reference_run_index=reference_run_index,
            candidate_run_index=candidate_run_index,
            step_index=step_index,
        )
        for reference_run_index, reference in enumerate(reference_results)
        for candidate_run_index, candidate in enumerate(candidate_results)
    ]
    return {
        "reference_num_runs": len(reference_results),
        "candidate_num_runs": len(candidate_results),
        "pairing": "cross-product",
        "num_pairs": len(comparisons),
        "comparisons": comparisons,
        "envelope": _summarize_comparison_envelope(comparisons),
    }


def _comparison_failures(
    comparisons: Sequence[dict[str, Any]],
    *,
    scope: str,
    require_exact: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for comparison in comparisons:
        pair = {
            "scope": scope,
            "reference_run_index": comparison["reference_run_index"],
            "candidate_run_index": comparison["candidate_run_index"],
        }
        trajectory = comparison["trajectory_metrics"]
        for field in (
            "timesteps_available",
            "timesteps_finite",
            "timesteps_match",
        ):
            if not trajectory.get(field, False):
                failures.append(pair | {"reason": field})

        for metrics in trajectory["per_step_metrics"]:
            location = pair | {"step_index": metrics["step_index"]}
            if not metrics["finite"]:
                failures.append(location | {"reason": "non_finite_trajectory"})
                continue
            if require_exact:
                if not metrics["exact_match"]:
                    failures.append(
                        location
                        | {
                            "reason": "repeatability_mismatch",
                            "max_abs": metrics["max_abs"],
                        }
                    )
                continue
            if not metrics["within_tolerance"]:
                failures.append(
                    location
                    | {
                        "reason": "outside_atol_rtol",
                        "max_abs": metrics["max_abs"],
                    }
                )
            if metrics["cosine_similarity"] < MODEL_QUALIFICATION_THRESHOLDS[
                "cosine_min"
            ]:
                failures.append(
                    location
                    | {
                        "reason": "cosine_below_minimum",
                        "cosine_similarity": metrics["cosine_similarity"],
                    }
                )
            if metrics["mae"] > MODEL_QUALIFICATION_THRESHOLDS["mae_max"]:
                failures.append(
                    location
                    | {
                        "reason": "mae_above_maximum",
                        "mae": metrics["mae"],
                    }
                )

        output_metrics = comparison["output_metrics"]["all_frames_metrics"]
        if not output_metrics["finite"]:
            failures.append(pair | {"reason": "non_finite_output_frames"})
        elif require_exact and not output_metrics["exact_match"]:
            failures.append(
                pair
                | {
                    "reason": "output_frame_repeatability_mismatch",
                    "max_abs": output_metrics["max_abs"],
                }
            )
    return failures


def evaluate_correctness_qualification(
    cross_variant_metrics: dict[str, Any],
    repeatability: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed over all trajectory steps, run pairs, and output finiteness."""
    failures = _comparison_failures(
        cross_variant_metrics.get("comparisons", []),
        scope="cross_variant",
        require_exact=False,
    )
    if not cross_variant_metrics.get("comparisons"):
        failures.append(
            {"scope": "cross_variant", "reason": "missing_comparisons"}
        )

    for variant in ("reference", "candidate"):
        summary = repeatability.get(variant)
        if not summary or not summary.get("available", False):
            failures.append(
                {"scope": f"{variant}_repeatability", "reason": "unavailable"}
            )
            continue
        failures.extend(
            _comparison_failures(
                summary.get("comparisons", []),
                scope=f"{variant}_repeatability",
                require_exact=True,
            )
        )
        if not summary.get("comparisons"):
            failures.append(
                {
                    "scope": f"{variant}_repeatability",
                    "reason": "missing_comparisons",
                }
            )

    return {
        "passed": not failures,
        "thresholds": dict(MODEL_QUALIFICATION_THRESHOLDS),
        "failures": failures,
    }


def evaluate_performance_qualification(
    performance: dict[str, Any],
) -> dict[str, Any]:
    """Qualify one execution order; callers must run both AB and BA."""
    speedup = performance.get("wall_median_speedup")
    failures = []
    if not isinstance(speedup, (int, float)) or not math.isfinite(speedup):
        failures.append({"reason": "missing_or_non_finite_wall_median_speedup"})
    elif speedup < MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]:
        failures.append(
            {
                "reason": "wall_median_speedup_below_minimum",
                "wall_median_speedup": speedup,
            }
        )
    return {
        "passed": not failures,
        "thresholds": {
            "speedup_min": MODEL_QUALIFICATION_THRESHOLDS["speedup_min"]
        },
        "failures": failures,
    }


def evaluate_candidate_backend_hit_qualification(
    per_run_hit_counts: Any,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if not isinstance(per_run_hit_counts, list) or not per_run_hit_counts:
        failures.append({"reason": "missing_candidate_hit_counts"})
    else:
        for run_index, hit_count in enumerate(per_run_hit_counts):
            if (
                isinstance(hit_count, bool)
                or not isinstance(hit_count, int)
                or hit_count <= 0
            ):
                failures.append(
                    {
                        "reason": "candidate_hit_count_not_positive",
                        "run_index": run_index,
                        "hit_count": hit_count,
                    }
                )
    return {
        "passed": not failures,
        "thresholds": {"candidate_hit_count_min_exclusive": 0},
        "failures": failures,
    }


def _with_candidate_backend_hit_qualification(
    qualification: dict[str, Any],
    per_run_hit_counts: Any,
) -> dict[str, Any]:
    hit_qualification = evaluate_candidate_backend_hit_qualification(
        per_run_hit_counts
    )
    hit_failures = [
        {"scope": "candidate_backend_hits"} | failure
        for failure in hit_qualification["failures"]
    ]
    return qualification | {
        "passed": qualification["passed"] and hit_qualification["passed"],
        "failures": qualification["failures"] + hit_failures,
        "candidate_backend_hits": hit_qualification,
    }


def extract_result_frames(result: Any) -> list[np.ndarray]:
    if result.frames is not None:
        return [np.asarray(frame) for frame in result.frames]

    sample = result.samples
    if sample is None:
        if result.output_file_path:
            output_path = Path(result.output_file_path)
            if not output_path.exists():
                raise ValueError(
                    "GenerationResult did not contain frames or samples, and its "
                    f"output_file_path does not exist: {output_path}"
                )
            if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return [np.asarray(iio.imread(output_path))]
            return [np.asarray(frame) for frame in iio.imiter(output_path)]
        raise ValueError(
            "GenerationResult did not contain frames, samples, or a readable output_file_path."
        )

    if isinstance(sample, torch.Tensor):
        tensor = sample.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(
                f"Unsupported tensor sample shape for frame extraction: {tuple(tensor.shape)}"
            )
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        frames = tensor.permute(1, 2, 3, 0).contiguous().numpy()
        return [frame for frame in frames]

    array = np.asarray(sample)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = array[None, ...]
        else:
            array = array[..., None]
    if array.ndim != 4:
        raise ValueError(
            f"Unsupported numpy sample shape for frame extraction: {tuple(array.shape)}"
        )
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return [frame for frame in array]


def build_server_kwargs(args: argparse.Namespace, *, variant: str) -> dict[str, Any]:
    component_paths = parse_component_overrides(
        getattr(args, f"{variant}_component_path") or []
    )
    component_attention_backends = parse_component_overrides(
        getattr(args, f"{variant}_component_attention_backend", None) or []
    )
    attention_backend_config = parse_json_object(
        getattr(args, f"{variant}_attention_backend_config", None),
        option_name=f"--{variant}-attention-backend-config",
    )
    transformer_path = getattr(args, f"{variant}_transformer_path")

    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "backend": args.backend,
        "num_gpus": args.num_gpus,
        "dit_cpu_offload": args.dit_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "pin_cpu_memory": args.pin_cpu_memory,
        "enable_cfg_parallel": args.enable_cfg_parallel,
        "ulysses_degree": args.ulysses_degree,
    }
    if args.master_port is not None:
        kwargs["master_port"] = args.master_port
    if args.sp_degree is not None:
        kwargs["sp_degree"] = args.sp_degree
    if transformer_path is not None:
        kwargs["transformer_weights_path"] = transformer_path
    if component_paths:
        kwargs["component_paths"] = component_paths
    if component_attention_backends:
        kwargs["component_attention_backends"] = component_attention_backends
    if attention_backend_config:
        # ServerArgs normalizes its string form into addict.Dict, which keeps
        # the attribute access used by existing sparse-attention request code.
        kwargs["attention_backend_config"] = json.dumps(
            attention_backend_config, sort_keys=True, separators=(",", ":")
        )
    attention_backend = getattr(args, f"{variant}_attention_backend")
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
    return kwargs


def build_sampling_kwargs(
    args: argparse.Namespace, *, output_dir: str | None = None
) -> dict[str, Any]:
    comparison_mode = getattr(args, "comparison_mode", "correctness")
    capture_trajectory = comparison_mode == "correctness"
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "return_frames": True,
        "return_trajectory_latents": capture_trajectory,
        "return_trajectory_decoded": (
            args.return_trajectory_decoded if capture_trajectory else False
        ),
        "save_output": output_dir is not None,
    }
    if output_dir is not None:
        kwargs["output_path"] = output_dir
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames
    if args.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    return kwargs


def _normalize_single_result(result: Any):
    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(
                f"Expected a single generation result, got {len(result)} results."
            )
        result = result[0]
    if result is None:
        raise RuntimeError("Generation returned no result.")
    return result


def _clear_diffusion_fp4_backend_caches() -> None:
    from sglang.multimodal_gen.runtime.layers.quantization import (
        modelopt_quant as diffusion_modelopt_quant,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    diffusion_modelopt_quant._get_fp4_gemm_op.cache_clear()
    current_platform.__class__.get_modelopt_fp4_gemm_op.cache_clear()
    current_platform.__class__.get_modelopt_flashinfer_fp4_backend.cache_clear()


@contextlib.contextmanager
def override_diffusion_fp4_backend(backend: str | None):
    env_name = "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"
    previous = os.environ.get(env_name)

    if backend is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = backend

    _clear_diffusion_fp4_backend_caches()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
        _clear_diffusion_fp4_backend_caches()


def _extract_total_duration_ms(result: Any) -> float | None:
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    total_duration_ms = metrics.get("total_duration_ms")
    if total_duration_ms is None:
        return None
    return float(total_duration_ms)


def _extract_wan_hybrid_hit_count(result: Any) -> int | None:
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    hit_count = metrics.get("wan_hybrid_hit_count")
    if isinstance(hit_count, bool) or not isinstance(hit_count, int):
        return None
    return hit_count


def _extract_generation_time_s(result: Any) -> float:
    """Return a populated end-to-end generation duration.

    ``DiffGenerator`` currently snapshots ``timer.duration`` before the timer
    context exits, so ``GenerationResult.generation_time`` can still be zero
    even though the scheduler metrics contain the completed request duration.
    Prefer a positive public field and otherwise use the scheduler's
    ``total_duration_ms`` measurement.
    """
    generation_time = float(result.generation_time)
    if generation_time > 0:
        return generation_time
    total_duration_ms = _extract_total_duration_ms(result)
    if total_duration_ms is None or total_duration_ms <= 0:
        raise ValueError(
            "Generation result contains neither a positive generation_time "
            "nor a positive metrics['total_duration_ms']."
        )
    return total_duration_ms / 1000.0


def _positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if (
        numerator is None
        or denominator is None
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0
        or denominator <= 0
    ):
        return None
    return numerator / denominator


def run_variant(
    *,
    server_kwargs: dict[str, Any],
    sampling_kwargs: dict[str, Any],
    fp4_gemm_backend: str | None,
    warmup_runs: int,
    measure_runs: int,
):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0.")
    if measure_runs <= 0:
        raise ValueError("measure_runs must be >= 1.")

    with override_diffusion_fp4_backend(fp4_gemm_backend):
        with DiffGenerator.from_pretrained(
            local_mode=True, **server_kwargs
        ) as generator:
            for _ in range(warmup_runs):
                _normalize_single_result(
                    generator.generate(sampling_params_kwargs=sampling_kwargs)
                )

            measured_results = []
            generation_times = []
            for _ in range(measure_runs):
                start_time = time.perf_counter()
                measured_results.append(
                    _normalize_single_result(
                        generator.generate(sampling_params_kwargs=sampling_kwargs)
                    )
                )
                generation_times.append(time.perf_counter() - start_time)

    final_result = measured_results[-1]
    peak_memories = [float(result.peak_memory_mb) for result in measured_results]
    total_duration_ms = [
        duration
        for duration in (
            _extract_total_duration_ms(result) for result in measured_results
        )
        if duration is not None
    ]
    wan_hybrid_hit_counts = [
        _extract_wan_hybrid_hit_count(result) for result in measured_results
    ]
    valid_wan_hybrid_hit_counts = [
        hit_count
        for hit_count in wan_hybrid_hit_counts
        if hit_count is not None
    ]

    return {
        "result": final_result,
        "_measured_results": measured_results,
        "fp4_gemm_backend": fp4_gemm_backend or "default",
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "generation_time_s": generation_times[-1],
        "avg_generation_time_s": sum(generation_times) / len(generation_times),
        "median_generation_time_s": statistics.median(generation_times),
        "per_run_generation_time_s": generation_times,
        "peak_memory_mb": peak_memories[-1],
        "max_peak_memory_mb": max(peak_memories) if peak_memories else 0.0,
        "per_run_peak_memory_mb": peak_memories,
        "total_duration_ms": total_duration_ms[-1] if total_duration_ms else None,
        "avg_total_duration_ms": (
            sum(total_duration_ms) / len(total_duration_ms)
            if total_duration_ms
            else None
        ),
        "median_total_duration_ms": (
            statistics.median(total_duration_ms) if total_duration_ms else None
        ),
        "per_run_total_duration_ms": total_duration_ms,
        "wan_hybrid_hit_count": (
            sum(valid_wan_hybrid_hit_counts)
            if len(valid_wan_hybrid_hit_counts) == len(wan_hybrid_hit_counts)
            else None
        ),
        "per_run_wan_hybrid_hit_count": wan_hybrid_hit_counts,
    }


def _to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-id",
        help=(
            "Optional model ID override passed to DiffGenerator.from_pretrained. "
            "Use this when --model-path points to a local directory whose name "
            "does not match a registered native SGLang model."
        ),
    )
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--master-port",
        type=int,
        help="Optional distributed master port used by both sequential variants.",
    )
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--sp-degree", type=int)
    parser.add_argument("--trajectory-step-index", type=int, default=-1)
    parser.add_argument(
        "--comparison-mode",
        choices=("correctness", "performance"),
        default="correctness",
        help=(
            "correctness captures all trajectory latents and computes all-pairs "
            "metrics; performance disables trajectory capture and reports timing only."
        ),
    )
    parser.add_argument(
        "--run-order",
        choices=("reference-first", "candidate-first"),
        default="reference-first",
        help=(
            "Execution order for order-bias checks. Semantic reference/candidate "
            "roles and reported speedup direction stay unchanged."
        ),
    )
    parser.add_argument(
        "--enforce-qualification",
        action="store_true",
        help=(
            "Exit unsuccessfully after writing the report when the fixed Wan "
            "model correctness/repeatability or E2E speedup gate fails."
        ),
    )
    parser.add_argument("--reference-transformer-path")
    parser.add_argument("--candidate-transformer-path")
    parser.add_argument(
        "--reference-attention-backend",
        help="Optional attention backend for the reference run.",
    )
    parser.add_argument(
        "--candidate-attention-backend",
        help="Optional attention backend for the candidate run.",
    )
    parser.add_argument(
        "--reference-attention-backend-config",
        help="Optional JSON object configuring the reference attention backend.",
    )
    parser.add_argument(
        "--candidate-attention-backend-config",
        help="Optional JSON object configuring the candidate attention backend.",
    )
    parser.add_argument(
        "--reference-component-attention-backend",
        action="append",
        default=[],
        help=(
            "Repeatable reference component attention override in the form "
            "component=backend."
        ),
    )
    parser.add_argument(
        "--candidate-component-attention-backend",
        action="append",
        default=[],
        help=(
            "Repeatable candidate component attention override in the form "
            "component=backend."
        ),
    )
    parser.add_argument(
        "--reference-fp4-gemm-backend",
        help=(
            "Optional NVFP4 GEMM backend override for the reference run, e.g. "
            "'flashinfer_trtllm'."
        ),
    )
    parser.add_argument(
        "--candidate-fp4-gemm-backend",
        help=(
            "Optional NVFP4 GEMM backend override for the candidate run, e.g. "
            "'flashinfer_trtllm'."
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    parser.add_argument(
        "--reference-component-path",
        action="append",
        default=[],
        help="Repeatable component override in the form component=path.",
    )
    parser.add_argument(
        "--candidate-component-path",
        action="append",
        default=[],
        help="Repeatable component override in the form component=path.",
    )
    parser.add_argument("--save-output-dir")
    parser.add_argument(
        "--return-trajectory-decoded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    save_root: Path | None = None
    if args.save_output_dir:
        save_root = Path(args.save_output_dir).expanduser().resolve()
        save_root.mkdir(parents=True, exist_ok=True)

    ref_server_kwargs = build_server_kwargs(args, variant="reference")
    cand_server_kwargs = build_server_kwargs(args, variant="candidate")

    ref_sampling_kwargs = build_sampling_kwargs(
        args,
        output_dir=str(save_root / "reference") if save_root else None,
    )
    cand_sampling_kwargs = build_sampling_kwargs(
        args,
        output_dir=str(save_root / "candidate") if save_root else None,
    )

    variant_calls = {
        "reference": {
            "server_kwargs": ref_server_kwargs,
            "sampling_kwargs": ref_sampling_kwargs,
            "fp4_gemm_backend": args.reference_fp4_gemm_backend,
        },
        "candidate": {
            "server_kwargs": cand_server_kwargs,
            "sampling_kwargs": cand_sampling_kwargs,
            "fp4_gemm_backend": args.candidate_fp4_gemm_backend,
        },
    }
    execution_order = (
        ("reference", "candidate")
        if args.run_order == "reference-first"
        else ("candidate", "reference")
    )
    runs: dict[str, dict[str, Any]] = {}
    for variant in execution_order:
        runs[variant] = run_variant(
            **variant_calls[variant],
            warmup_runs=args.warmup_runs,
            measure_runs=args.measure_runs,
        )
    reference_run = runs["reference"]
    candidate_run = runs["candidate"]
    reference = reference_run["result"]
    candidate = candidate_run["result"]

    result = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "comparison_mode": args.comparison_mode,
        "run_order": args.run_order,
        "server_kwargs": {
            "reference": ref_server_kwargs,
            "candidate": cand_server_kwargs,
        },
        "backend_overrides": {
            "reference_attention_backend": args.reference_attention_backend
            or "default",
            "candidate_attention_backend": args.candidate_attention_backend
            or "default",
            "reference_fp4_gemm_backend": reference_run["fp4_gemm_backend"],
            "candidate_fp4_gemm_backend": candidate_run["fp4_gemm_backend"],
        },
        "sampling_kwargs": {
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
        },
        "reference_generation": {
            key: value
            for key, value in reference_run.items()
            if key != "result" and not key.startswith("_")
        }
        | {"output_file_path": reference.output_file_path},
        "candidate_generation": {
            key: value
            for key, value in candidate_run.items()
            if key != "result" and not key.startswith("_")
        }
        | {"output_file_path": candidate.output_file_path},
        "performance": {
            "wall_median_speedup": _positive_ratio(
                reference_run["median_generation_time_s"],
                candidate_run["median_generation_time_s"],
            ),
            "scheduler_median_speedup": _positive_ratio(
                reference_run["median_total_duration_ms"],
                candidate_run["median_total_duration_ms"],
            ),
        },
    }

    if args.comparison_mode == "correctness":
        result["repeatability"] = {
            "reference": summarize_run_repeatability(
                reference_run["_measured_results"],
                step_index=args.trajectory_step_index,
            ),
            "candidate": summarize_run_repeatability(
                candidate_run["_measured_results"],
                step_index=args.trajectory_step_index,
            ),
        }
        result["cross_variant_metrics"] = summarize_cross_variant_metrics(
            reference_run["_measured_results"],
            candidate_run["_measured_results"],
            step_index=args.trajectory_step_index,
        )
        result["qualification"] = evaluate_correctness_qualification(
            result["cross_variant_metrics"], result["repeatability"]
        )
    else:
        result["qualification"] = evaluate_performance_qualification(
            result["performance"]
        )
    result["qualification"] = _with_candidate_backend_hit_qualification(
        result["qualification"],
        result["candidate_generation"]["per_run_wan_hybrid_hit_count"],
    )

    output_json.write_text(
        json.dumps(_to_jsonable(result), indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "output_json": str(output_json),
        "comparison_mode": args.comparison_mode,
        "run_order": args.run_order,
        "reference_median_generation_time_s": result["reference_generation"][
            "median_generation_time_s"
        ],
        "candidate_median_generation_time_s": result["candidate_generation"][
            "median_generation_time_s"
        ],
        "candidate_wan_hybrid_hit_count": result["candidate_generation"][
            "wan_hybrid_hit_count"
        ],
        "candidate_per_run_wan_hybrid_hit_count": result["candidate_generation"][
            "per_run_wan_hybrid_hit_count"
        ],
        **result["performance"],
        "qualification_passed": result["qualification"]["passed"],
        "qualification_failures": result["qualification"]["failures"],
    }
    if args.comparison_mode == "correctness":
        summary["cross_variant_envelope"] = result["cross_variant_metrics"][
            "envelope"
        ]
    print(json.dumps(summary, indent=2))
    if args.enforce_qualification and not result["qualification"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
