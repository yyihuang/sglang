"""Immutable inputs and gates for the public GDN qualification campaign.

The campaign runner writes raw observations.  ``audit.py`` independently
recomputes every acceptance metric from those observations and refuses partial,
duplicated, or provenance-drifted evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA = "gdn-public-qualification-result-v2"
PLAN_SCHEMA = "gdn-public-qualification-plan-v2"
ROUTE_ARTIFACT_SCHEMA = "gdn-noncp-final-sglang-route-artifact-v1"
KL_DISTRIBUTION_SCHEMA = "gdn-full-vocabulary-logprob-distribution-v1"

SGLANG_INTEGRATION_COMMIT = "d1aeb7785547d6de57ff9b199662726664af8099"
SGLANG_INTEGRATION_TREE = "5e9a389267abe5e7354f7730dc022a2d0f4b0e3d"
FLASHINFER_COMMIT = "d65b164713541652e02518e9be1cfd20350ddc7e"
FLASHINFER_TREE = "8f734e778a0630b78f8dac0c420f9193929165cc"
FLASHINFER_PARENT_COMMIT = "93151678bcd020310aac1b764eb83a994de957dd"
SOURCE_COMMIT = "940094ba84ec091f778db8966df1a2b94ff1c99a"
SOURCE_TREE = "977cc62eb1f0f4781c6923f49fd542151949940b"

CONTAINER_IMAGE = "nvcr.io/nvidia/sglang:26.07-py3"
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_REVISION = "c5f5f263bdd5cc134092897864e8905d8fe7b928"

HASHES = {
    "flashinfer_bundle_sha256": "1cbdb1f7be86eadfa86175fac067b7c71f2a9d18d1e505a8f12b4c3f57584517",
    "source_bundle_sha256": "48f3ff6ba5d35f84c422a3f0be3d00b80be4253e0ef136be7661e0435d7ba603",
    "public_export_archive_sha256": "51efd0b6bb9c86770c3b74cfdc8e8ec0f0c675b2ff83dbbde3d8d403488a909e",
    "core_manifest_sha256": "e83d8af2f1e10f8f075d9c1e92c87315f63851fe4651374ce27b9b0eb20d81e5",
    "overlay_manifest_sha256": "f915892b1a47513b0e9bbe38432f701d2b57198c0df9a95e6199676859961185",
    "kernel_sha256": "fde8e19ea3e717c0c966f8bd482f72e155c20bf644601153140aefb1b969b7fa",
    "exporter_sha256": "93dfd0d8bb15510e007501ebe68fce11d10e663de12903c5527bceabc555ff1a",
    "input_delivery_manifest_sha256": "f9b6aac29e058694edccfd6379e738db526c12e37a40f0773fe7e65af63024ef",
    "source_authority_sha256": "14b3175be84dcc57c681300868ff97e2992a5da7184df1b002f233b9ffef1c09",
    "model_manifest_sha256": "49f46f7e1b93abad35e295348cc8e1477b3df1a0c597a97791ac7d7a6d7b0a06",
    "model_stage_receipt_sha256": "374dc629e32e2a1c5d50972fd38893a8f627d94e71450ef8c86784a42359f2b8",
    "gsm8k_dataset_sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "gsm8k_manifest_sha256": "0b6bdeda8b61ffb2d25e83c78191ba94ed1ba295eaa558e8a72865a6bcc5a5a5",
    "gsm8k_prompt_ids_sha256": "6f2e88c9df2642a658293f7c0a7dd30ea8320414d2b9204b6130b10a06035e8a",
    "longbench_dataset_sha256": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
    "longbench_manifest_sha256": "c47e946834ae2dcff7aa7a372dbec0a55f6b94ae68308213f8d6d55f189a2342",
    "longbench_first32_ids_sha256": "5f957f68a8f105b20c40fa5d49b8237c49b488b7b3298a8fb43535e41ace2033",
    "longbench_first48_ids_sha256": "36eb64beb3957576a28bdde7a51698912f01b8ae9c6a0353a8413a322aac47b7",
}

WORKLOADS = {
    "longbench_first32": HASHES["longbench_first32_ids_sha256"],
    "longbench_first48": HASHES["longbench_first48_ids_sha256"],
}

PROMPT_COUNT = 1314
GSM8K_SHOTS = 5
MIN_SCORE = 0.93
MAX_KL_EXCLUSIVE = 0.0035
KL_SAMPLE_COUNT = 48
KL_METRIC = "full_vocabulary_forward_kl_p_baseline_q_candidate"
KL_DIRECTION = "P_baseline||Q_candidate"
KL_POSITION_AGGREGATION = "arithmetic_mean_over_scored_positions_per_sample"
KL_SAMPLE_AGGREGATION = "maximum_across_48_sealed_samples"
KL_TOKEN_ID_ORDER = "ascending_integer_ids_0_to_vocab_size_minus_1"
KL_VOCAB_CHUNK_SIZE = 8192
KL_NORMALIZATION_ATOL = 5e-4
KL_NEGATIVE_ATOL = 1e-6
TP_SIZE = 4
TP_RANKS = [0, 1, 2, 3]
MTP_PROBE_PROMPT_INDEX = 0
MTP_PROBE_MAX_NEW_TOKENS = 8
MTP_SPECULATIVE_NUM_STEPS = 3
MTP_SPECULATIVE_EAGLE_TOPK = 1
MTP_SPECULATIVE_NUM_DRAFT_TOKENS = 4
EXACT_T4_ROUTE = (
    "flashinfer.gdn_decode.noncp.indexed_bf16_verify_t4.tile16_fullwarp"
)
# These exact exported contract rows are the routes exercised by the pinned
# SGLang TP4 campaign.  The route-artifact producer resolves their route IDs
# from the authenticated final export instead of accepting route names from a
# caller.
SGLANG_ROUTE_CONTRACT_ROWS = {
    "prefill": (
        ("prefill_focus", "correctness_sglang_tp4_bf16_indexed_b5_s64"),
        (
            "prefill_focus",
            "correctness_sglang_tp4_bf16_indexed_checkpoint_b7_t421",
        ),
    ),
    "decode": (
        ("decode_bf16_serving", "bf16_sglang_qwen_tp4_decode_t1"),
        ("decode_bf16_serving", "bf16_sglang_qwen_tp4_verify_t4"),
    ),
}
ABBA_ORDER = ["baseline", "candidate", "candidate", "baseline"] * 4
OBSERVATIONS_PER_ARM_PER_WORKLOAD = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 2026083001

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTE_RE = {
    "prefill": re.compile(
        r"^flashinfer\.gdn_prefill\.noncp\.[a-z0-9_.-]+$"
    ),
    "decode": re.compile(r"^flashinfer\.gdn_decode\.noncp\.[a-z0-9_.-]+$"),
}


class QualificationError(ValueError):
    """A receipt violates the immutable qualification contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _finite_positive(value: object, label: str) -> float:
    require(isinstance(value, (int, float)), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0, f"{label} must be finite and > 0")
    return result


def _geomean(values: Sequence[float]) -> float:
    require(bool(values), "geomean input must not be empty")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    require(bool(sorted_values), "percentile input must not be empty")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def expected_provenance() -> dict[str, object]:
    return {
        "sglang_integration_commit": SGLANG_INTEGRATION_COMMIT,
        "sglang_integration_tree": SGLANG_INTEGRATION_TREE,
        "flashinfer_commit": FLASHINFER_COMMIT,
        "flashinfer_tree": FLASHINFER_TREE,
        "flashinfer_parent_commit": FLASHINFER_PARENT_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "container_image": CONTAINER_IMAGE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **HASHES,
    }


def validate_provenance(provenance: Mapping[str, object]) -> None:
    for key, expected in expected_provenance().items():
        require(provenance.get(key) == expected, f"provenance mismatch for {key}")
    qualification_commit = provenance.get("qualification_commit")
    require(
        isinstance(qualification_commit, str)
        and bool(_FULL_SHA_RE.fullmatch(qualification_commit)),
        "qualification_commit must be a full lowercase Git SHA",
    )
    require(
        provenance.get("compute_capability") == [10, 3],
        "final campaign must run directly on compute capability 10.3",
    )
    require(provenance.get("tp_size") == TP_SIZE, "tp_size must be 4")
    require(provenance.get("tp_ranks") == TP_RANKS, "tp_ranks must be [0,1,2,3]")
    require(
        isinstance(provenance.get("gpu_name"), str)
        and bool(provenance["gpu_name"].strip()),
        "gpu_name is required",
    )
    require(
        isinstance(provenance.get("cuda_version"), str)
        and bool(provenance["cuda_version"].strip()),
        "cuda_version is required",
    )


def _validate_prompt_rows(rows: object, arm: str) -> float:
    require(isinstance(rows, list), f"{arm} prompts must be a list")
    require(len(rows) == PROMPT_COUNT, f"{arm} must contain exactly 1314 prompts")
    question_indices = []
    source_indices = []
    scores = []
    for position, row in enumerate(rows):
        require(isinstance(row, Mapping), f"{arm} prompt {position} must be an object")
        question_indices.append(row.get("question_index"))
        source_indices.append(row.get("source_row_index"))
        require(row.get("request_count") == 1, f"{arm} prompt {position} was not requested exactly once")
        require(row.get("correct") in (True, False), f"{arm} prompt {position} has invalid correctness")
        scores.append(float(row["correct"]))
    require(question_indices == list(range(PROMPT_COUNT)), f"{arm} question indices must be exactly 0..1313 in order")
    require(source_indices == list(range(GSM8K_SHOTS, GSM8K_SHOTS + PROMPT_COUNT)), f"{arm} source rows must be exactly 5..1318 in order")
    require(len(set(question_indices)) == PROMPT_COUNT, f"{arm} question indices are duplicated")
    require(len(set(source_indices)) == PROMPT_COUNT, f"{arm} source rows are duplicated")
    return sum(scores) / PROMPT_COUNT


def _resolve_evidence_path(root: Path, value: object, label: str) -> Path:
    require(isinstance(value, str) and bool(value), f"{label} path is required")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    require(resolved.is_relative_to(resolved_root), f"{label} path escapes the evidence root")
    require(resolved.is_file(), f"missing {label} file {relative}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_kl_manifest(spec: object, evidence_root: Path, arm: str) -> tuple[dict, Path]:
    require(isinstance(spec, Mapping), f"{arm} KL manifest reference must be an object")
    path = _resolve_evidence_path(evidence_root, spec.get("path"), f"{arm} KL manifest")
    expected_sha = spec.get("sha256")
    require(isinstance(expected_sha, str) and bool(_SHA256_RE.fullmatch(expected_sha)), f"{arm} KL manifest SHA256 is required")
    require(_file_sha256(path) == expected_sha, f"{arm} KL manifest SHA256 mismatch")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{arm} KL manifest is not valid JSON") from exc
    require(isinstance(manifest, dict), f"{arm} KL manifest must be an object")
    require(manifest.get("schema") == KL_DISTRIBUTION_SCHEMA, f"{arm} KL distribution schema differs")
    require(manifest.get("arm") == arm, f"{arm} KL manifest arm differs")
    return manifest, path.parent


def _validate_kl_manifest_identity(manifest: Mapping[str, object], arm: str) -> None:
    require(manifest.get("sample_count") == KL_SAMPLE_COUNT, f"{arm} KL sample count must be 48")
    require(manifest.get("input_ids_sha256") == HASHES["longbench_first48_ids_sha256"], f"{arm} KL input IDs hash mismatch")
    require(manifest.get("token_id_order") == KL_TOKEN_ID_ORDER, f"{arm} KL token-ID order differs")
    require(manifest.get("dtype") == "float32", f"{arm} KL dtype must be float32")
    require(manifest.get("byte_order") == "little", f"{arm} KL byte order must be little")
    require(manifest.get("normalization_atol") == KL_NORMALIZATION_ATOL, f"{arm} KL normalization tolerance differs")
    require(manifest.get("vocab_chunk_size") == KL_VOCAB_CHUNK_SIZE, f"{arm} KL vocabulary chunk size differs")
    require(manifest.get("model_manifest_sha256") == HASHES["model_manifest_sha256"], f"{arm} KL model manifest hash differs")
    for key in ("model_path", "tokenizer_path"):
        require(isinstance(manifest.get(key), str) and bool(manifest[key]), f"{arm} KL {key} is required")
    vocab_size = manifest.get("vocab_size")
    require(type(vocab_size) is int and vocab_size > 1, f"{arm} KL vocab_size must be an integer > 1")
    rows = manifest.get("records")
    require(isinstance(rows, list) and len(rows) == KL_SAMPLE_COUNT, f"{arm} KL must contain exactly 48 records")
    require([row.get("sample_index") for row in rows if isinstance(row, Mapping)] == list(range(KL_SAMPLE_COUNT)), f"{arm} KL sample indices must be exactly 0..47 in order")


def _read_f32_matrix(path: Path, expected_values: int, label: str):
    expected_bytes = expected_values * 4
    require(path.stat().st_size == expected_bytes, f"{label} byte count differs")
    raw = path.read_bytes()
    try:
        import numpy as np
    except ModuleNotFoundError:
        require(expected_values <= 1_000_000, "NumPy is required to audit production KL distribution shards")
        return [value[0] for value in struct.iter_unpack("<f", raw)]
    return np.frombuffer(raw, dtype="<f4").astype(np.float64, copy=False)


def _row_stats(baseline, candidate) -> tuple[float, float, float]:
    """Return baseline mass, candidate mass, and full-vocabulary P||Q."""

    try:
        import numpy as np
    except ModuleNotFoundError:
        baseline_values = [float(value) for value in baseline]
        candidate_values = [float(value) for value in candidate]
        require(all(math.isfinite(value) for value in baseline_values + candidate_values), "KL logprobs must be finite")
        require(all(value <= 1e-6 for value in baseline_values + candidate_values), "KL logprobs must not be positive")
        probabilities = [math.exp(value) for value in baseline_values]
        return (
            math.fsum(probabilities),
            math.fsum(math.exp(value) for value in candidate_values),
            math.fsum(probability * (p_value - q_value) for probability, p_value, q_value in zip(probabilities, baseline_values, candidate_values)),
        )
    require(bool(np.isfinite(baseline).all()) and bool(np.isfinite(candidate).all()), "KL logprobs must be finite")
    require(float(np.max(baseline)) <= 1e-6 and float(np.max(candidate)) <= 1e-6, "KL logprobs must not be positive")
    baseline_probability = np.exp(baseline)
    return (
        float(np.sum(baseline_probability, dtype=np.float64)),
        float(np.sum(np.exp(candidate), dtype=np.float64)),
        float(np.sum(baseline_probability * (baseline - candidate), dtype=np.float64)),
    )


def _validate_kl_record(row: object, sample_index: int, vocab_size: int, arm: str) -> tuple[int, list[Mapping[str, object]]]:
    require(isinstance(row, Mapping), f"{arm} KL sample {sample_index} must be an object")
    require(row.get("sample_index") == sample_index, f"{arm} KL sample {sample_index} index differs")
    output_ids = row.get("output_ids")
    require(isinstance(output_ids, list) and bool(output_ids) and all(type(token) is int and 0 <= token < vocab_size for token in output_ids), f"{arm} KL sample {sample_index} output IDs are invalid")
    require(row.get("position_count") == len(output_ids), f"{arm} KL sample {sample_index} position count differs")
    require(row.get("output_ids_sha256") == _json_sha256(output_ids), f"{arm} KL sample {sample_index} output ID hash differs")
    require(isinstance(row.get("input_ids_sha256"), str) and bool(_SHA256_RE.fullmatch(row["input_ids_sha256"])), f"{arm} KL sample {sample_index} input ID hash is required")
    shards = row.get("shards")
    require(isinstance(shards, list) and bool(shards), f"{arm} KL sample {sample_index} shards are required")
    cursor = 0
    for shard_index, shard in enumerate(shards):
        require(isinstance(shard, Mapping), f"{arm} KL sample {sample_index} shard {shard_index} must be an object")
        start = shard.get("token_start")
        end = shard.get("token_end")
        require(type(start) is int and type(end) is int and start == cursor and start < end <= vocab_size, f"{arm} KL sample {sample_index} shard {shard_index} vocabulary coverage differs")
        require(end - start <= KL_VOCAB_CHUNK_SIZE, f"{arm} KL sample {sample_index} shard {shard_index} is too wide")
        require(shard.get("shape") == [len(output_ids), end - start], f"{arm} KL sample {sample_index} shard {shard_index} shape differs")
        require(shard.get("byte_count") == len(output_ids) * (end - start) * 4, f"{arm} KL sample {sample_index} shard {shard_index} byte count differs")
        require(isinstance(shard.get("sha256"), str) and bool(_SHA256_RE.fullmatch(shard["sha256"])), f"{arm} KL sample {sample_index} shard {shard_index} SHA256 is required")
        cursor = end
    require(cursor == vocab_size, f"{arm} KL sample {sample_index} does not cover the full vocabulary")
    return len(output_ids), shards


def _recompute_kl(kl: Mapping[str, object], evidence_root: Path | None) -> tuple[list[float], float]:
    """Recompute exact full-vocabulary D_KL(P_baseline || Q_candidate).

    Each scored teacher-forced output position is normalized over every token
    ID. Position KL values are averaged within a sample; the strict campaign
    gate is the maximum of those 48 sealed sample means.
    """

    require(evidence_root is not None, "an evidence root is required for KL distribution manifests")
    baseline_manifest, baseline_root = _load_kl_manifest(kl.get("baseline_manifest"), evidence_root, "baseline")
    candidate_manifest, candidate_root = _load_kl_manifest(kl.get("candidate_manifest"), evidence_root, "candidate")
    _validate_kl_manifest_identity(baseline_manifest, "baseline")
    _validate_kl_manifest_identity(candidate_manifest, "candidate")
    for key in ("input_ids_sha256", "model_manifest_sha256", "model_path", "tokenizer_path", "vocab_size", "token_id_order", "dtype", "byte_order", "normalization_atol", "vocab_chunk_size"):
        require(candidate_manifest.get(key) == baseline_manifest.get(key), f"KL baseline/candidate {key} alignment differs")
    baseline_spec = kl["baseline_manifest"]
    require(candidate_manifest.get("reference_manifest_sha256") == baseline_spec.get("sha256"), "candidate KL reference manifest hash differs")

    vocab_size = baseline_manifest["vocab_size"]
    baseline_rows = baseline_manifest["records"]
    candidate_rows = candidate_manifest["records"]
    sample_means = []
    max_position_kl = 0.0
    observed_paths: set[Path] = set()
    for sample_index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
        position_count, baseline_shards = _validate_kl_record(baseline_row, sample_index, vocab_size, "baseline")
        candidate_position_count, candidate_shards = _validate_kl_record(candidate_row, sample_index, vocab_size, "candidate")
        require(candidate_position_count == position_count, f"KL sample {sample_index} position alignment differs")
        for key in ("input_ids_sha256", "output_ids_sha256", "output_ids", "position_count"):
            require(candidate_row.get(key) == baseline_row.get(key), f"KL sample {sample_index} {key} alignment differs")
        require(len(candidate_shards) == len(baseline_shards), f"KL sample {sample_index} shard count differs")
        baseline_mass = [0.0] * position_count
        candidate_mass = [0.0] * position_count
        position_kl = [0.0] * position_count
        for shard_index, (baseline_shard, candidate_shard) in enumerate(zip(baseline_shards, candidate_shards)):
            for key in ("token_start", "token_end", "shape", "byte_count"):
                require(candidate_shard.get(key) == baseline_shard.get(key), f"KL sample {sample_index} shard {shard_index} {key} alignment differs")
            width = baseline_shard["token_end"] - baseline_shard["token_start"]
            matrices = []
            for arm, shard, root in (("baseline", baseline_shard, baseline_root), ("candidate", candidate_shard, candidate_root)):
                path = _resolve_evidence_path(root, shard.get("path"), f"{arm} KL shard")
                require(path not in observed_paths, f"KL shard path is reused: {path.name}")
                observed_paths.add(path)
                require(_file_sha256(path) == shard.get("sha256"), f"{arm} KL sample {sample_index} shard {shard_index} SHA256 mismatch")
                matrices.append(_read_f32_matrix(path, position_count * width, f"{arm} KL sample {sample_index} shard {shard_index}"))
            baseline_values, candidate_values = matrices
            for position in range(position_count):
                start = position * width
                end = start + width
                p_mass, q_mass, kl_value = _row_stats(baseline_values[start:end], candidate_values[start:end])
                baseline_mass[position] += p_mass
                candidate_mass[position] += q_mass
                position_kl[position] += kl_value
        for position, (p_mass, q_mass, kl_value) in enumerate(zip(baseline_mass, candidate_mass, position_kl)):
            require(abs(p_mass - 1.0) <= KL_NORMALIZATION_ATOL, f"KL sample {sample_index} position {position} baseline probability mass {p_mass:.9f} is not normalized")
            require(abs(q_mass - 1.0) <= KL_NORMALIZATION_ATOL, f"KL sample {sample_index} position {position} candidate probability mass {q_mass:.9f} is not normalized")
            require(math.isfinite(kl_value), f"KL sample {sample_index} position {position} is not finite")
            require(kl_value >= -KL_NEGATIVE_ATOL, f"KL sample {sample_index} position {position} is negative")
            position_kl[position] = max(0.0, kl_value)
        sample_means.append(math.fsum(position_kl) / position_count)
        max_position_kl = max(max_position_kl, max(position_kl))
    return sample_means, max_position_kl


def validate_accuracy(accuracy: Mapping[str, object], evidence_root: Path | None = None) -> dict[str, object]:
    arms = accuracy.get("arms")
    require(isinstance(arms, Mapping) and set(arms) == {"baseline", "candidate"}, "accuracy arms must be exactly baseline and candidate")
    scores = {}
    for arm in ("baseline", "candidate"):
        arm_result = arms[arm]
        require(isinstance(arm_result, Mapping), f"accuracy {arm} must be an object")
        score = _validate_prompt_rows(arm_result.get("prompts"), arm)
        reported = arm_result.get("score")
        require(isinstance(reported, (int, float)) and math.isclose(float(reported), score, rel_tol=0, abs_tol=1e-15), f"{arm} reported score does not match raw prompts")
        require(score >= MIN_SCORE, f"{arm} score {score:.9f} is below 0.93")
        scores[arm] = score
    require(scores["candidate"] >= scores["baseline"], "candidate accuracy dropped relative to baseline")
    kl = accuracy.get("kl")
    require(isinstance(kl, Mapping), "accuracy.kl must be an object")
    require(kl.get("metric") == KL_METRIC, f"KL metric must be {KL_METRIC}")
    require(kl.get("direction") == KL_DIRECTION, f"KL direction must be {KL_DIRECTION}")
    require(kl.get("position_aggregation") == KL_POSITION_AGGREGATION, f"KL position aggregation must be {KL_POSITION_AGGREGATION}")
    require(kl.get("sample_aggregation") == KL_SAMPLE_AGGREGATION, f"KL sample aggregation must be {KL_SAMPLE_AGGREGATION}")
    sample_values, max_position_kl = _recompute_kl(kl, evidence_root)
    mean_kl = sum(sample_values) / len(sample_values)
    max_kl = max(sample_values)
    for key, computed in (("mean_sample_kl", mean_kl), ("max_sample_kl", max_kl), ("max_position_kl", max_position_kl)):
        reported = kl.get(key)
        require(isinstance(reported, (int, float)) and math.isclose(float(reported), computed, rel_tol=0, abs_tol=1e-12), f"reported {key} does not match full-vocabulary distributions")
    require(max_kl < MAX_KL_EXCLUSIVE, f"maximum sample full-vocabulary KL {max_kl:.9f} is not < 0.0035")
    return {**scores, "kl_metric": KL_METRIC, "kl_direction": KL_DIRECTION, "mean_sample_kl": mean_kl, "max_sample_kl": max_kl, "max_position_kl": max_position_kl}


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_mtp_probe(probe: Mapping[str, object]) -> dict[str, object]:
    arms = probe.get("arms")
    require(
        isinstance(arms, Mapping) and set(arms) == {"baseline", "candidate"},
        "MTP probe arms must be exactly baseline and candidate",
    )
    result = {}
    for arm, backend in (("baseline", "triton"), ("candidate", "flashinfer")):
        row = arms[arm]
        require(isinstance(row, Mapping), f"MTP probe {arm} must be an object")
        require(row.get("arm") == arm, f"MTP probe {arm} arm identity differs")
        require(
            row.get("input_ids_sha256") == HASHES["longbench_first48_ids_sha256"],
            f"MTP probe {arm} input IDs hash mismatch",
        )
        require(
            row.get("prompt_index") == MTP_PROBE_PROMPT_INDEX,
            f"MTP probe {arm} prompt index differs",
        )
        require(
            row.get("request_count") == 1,
            f"MTP probe {arm} was not requested exactly once",
        )
        require(
            row.get("sampling_params")
            == {
                "temperature": 0.0,
                "max_new_tokens": MTP_PROBE_MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            f"MTP probe {arm} sampling parameters differ",
        )
        server_config = row.get("server_config")
        require(
            server_config
            == {
                "tp_size": TP_SIZE,
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": MTP_SPECULATIVE_NUM_STEPS,
                "speculative_eagle_topk": MTP_SPECULATIVE_EAGLE_TOPK,
                "speculative_num_draft_tokens": MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
                "linear_attn_prefill_backend": backend,
                "linear_attn_decode_backend": backend,
                "linear_attn_verify_backend": backend,
            },
            f"MTP probe {arm} server configuration differs",
        )
        output_ids = row.get("output_ids")
        require(
            isinstance(output_ids, list)
            and len(output_ids) == MTP_PROBE_MAX_NEW_TOKENS
            and all(type(token) is int for token in output_ids),
            f"MTP probe {arm} must return exactly {MTP_PROBE_MAX_NEW_TOKENS} integer output IDs",
        )
        output_ids_sha256 = row.get("output_ids_sha256")
        require(
            isinstance(output_ids_sha256, str)
            and output_ids_sha256 == _json_sha256(output_ids),
            f"MTP probe {arm} output IDs hash mismatch",
        )
        runtime = _finite_positive(
            row.get("measured_runtime_seconds"),
            f"MTP probe {arm} measured runtime",
        )
        result[arm] = {
            "output_ids_sha256": output_ids_sha256,
            "measured_runtime_seconds": runtime,
        }
    return result


def _validate_route_names(route_kind: str, routes: object) -> list[str]:
    require(isinstance(routes, list) and bool(routes), f"candidate {route_kind} routes must be a non-empty list")
    require(routes == sorted(set(routes)), f"candidate {route_kind} routes must be sorted and unique")
    for route in routes:
        require(isinstance(route, str) and bool(_ROUTE_RE[route_kind].fullmatch(route)), f"invalid {route_kind} non-CP route {route!r}")
    return routes


def validate_routes(routes: Mapping[str, object]) -> dict[str, object]:
    expected = routes.get("expected_candidate_routes")
    require(isinstance(expected, Mapping) and set(expected) == {"prefill", "decode"}, "expected candidate routes must contain prefill and decode")
    expected_routes = {kind: _validate_route_names(kind, expected[kind]) for kind in ("prefill", "decode")}
    arms = routes.get("arms")
    require(isinstance(arms, Mapping) and set(arms) == {"baseline", "candidate"}, "route arms must be exactly baseline and candidate")
    for arm in ("baseline", "candidate"):
        rank_rows = arms[arm]
        require(isinstance(rank_rows, list) and len(rank_rows) == TP_SIZE, f"{arm} must report exactly four TP ranks")
        require([row.get("rank") for row in rank_rows if isinstance(row, Mapping)] == TP_RANKS, f"{arm} ranks must be exactly 0..3 in order")
        for row in rank_rows:
            require(isinstance(row, Mapping), f"{arm} rank row must be an object")
            prefill = row.get("prefill_routes")
            decode = row.get("decode_routes")
            require(isinstance(prefill, list) and isinstance(decode, list), f"{arm} rank {row.get('rank')} route fields must be lists")
            require(row.get("fallback_count") == 0, f"{arm} rank {row.get('rank')} used a fallback")
            require(row.get("route_error_count") == 0, f"{arm} rank {row.get('rank')} had route errors")
            observations = row.get("route_observations")
            require(isinstance(observations, list), f"{arm} rank {row.get('rank')} route observations must be a list")
            marker_count = row.get("marker_count")
            require(isinstance(marker_count, int) and marker_count >= len(observations), f"{arm} rank {row.get('rank')} marker count is incomplete")
            observed_t4_routes = []
            for observation in observations:
                require(isinstance(observation, Mapping), f"{arm} rank {row.get('rank')} route observation must be an object")
                phase = observation.get("phase")
                route = observation.get("route")
                token_width = observation.get("t")
                require(phase in {"prefill", "decode"}, f"{arm} rank {row.get('rank')} route phase is invalid")
                require(isinstance(token_width, int) and token_width > 0, f"{arm} rank {row.get('rank')} route token width is invalid")
                require(observation.get("gates_present") is True, f"{arm} rank {row.get('rank')} route marker lacks gate authority")
                require(route in (prefill if phase == "prefill" else decode), f"{arm} rank {row.get('rank')} route observation differs from route sets")
                if phase == "decode" and token_width == 4:
                    observed_t4_routes.append(route)
            if arm == "candidate":
                require(prefill == expected_routes["prefill"], f"candidate rank {row.get('rank')} prefill routes differ")
                require(decode == expected_routes["decode"], f"candidate rank {row.get('rank')} decode routes differ")
                require(sorted(set(observed_t4_routes)) == [EXACT_T4_ROUTE], f"candidate rank {row.get('rank')} did not prove the exact optimized T=4 route")
            else:
                require(prefill == [] and decode == [], f"baseline rank {row.get('rank')} recorded optimized GDN routes")
                require(marker_count == 0 and observations == [], f"baseline rank {row.get('rank')} optimized-route marker count is nonzero")
    return {**expected_routes, "exact_t4_route": EXACT_T4_ROUTE}


def _workload_block_ratios(workload: Mapping[str, object]) -> list[float]:
    workload_id = workload.get("workload_id")
    require(workload_id in WORKLOADS, f"unknown workload {workload_id!r}")
    require(workload.get("input_ids_sha256") == WORKLOADS[workload_id], f"workload {workload_id} input IDs hash mismatch")
    observations = workload.get("observations")
    require(isinstance(observations, list) and len(observations) == len(ABBA_ORDER), f"workload {workload_id} must contain exactly 16 observations")
    require([row.get("sequence_index") for row in observations if isinstance(row, Mapping)] == list(range(len(ABBA_ORDER))), f"workload {workload_id} sequence indices must be exactly 0..15")
    require([row.get("arm") for row in observations if isinstance(row, Mapping)] == ABBA_ORDER, f"workload {workload_id} order must be ABBA repeated four times")
    counts = {arm: sum(row.get("arm") == arm for row in observations) for arm in ("baseline", "candidate")}
    require(counts == {"baseline": OBSERVATIONS_PER_ARM_PER_WORKLOAD, "candidate": OBSERVATIONS_PER_ARM_PER_WORKLOAD}, f"workload {workload_id} must contain eight observations per arm")
    throughputs = []
    for index, row in enumerate(observations):
        require(isinstance(row, Mapping), f"workload {workload_id} observation {index} must be an object")
        throughputs.append(_finite_positive(row.get("throughput_tokens_per_second"), f"workload {workload_id} observation {index} throughput"))
        _finite_positive(row.get("measured_runtime_seconds"), f"workload {workload_id} observation {index} measured runtime")
    block_ratios = []
    for offset in range(0, len(throughputs), 4):
        baseline = [throughputs[offset], throughputs[offset + 3]]
        candidate = [throughputs[offset + 1], throughputs[offset + 2]]
        block_ratios.append(_geomean(candidate) / _geomean(baseline))
    return block_ratios


def validate_performance(performance: Mapping[str, object]) -> dict[str, object]:
    require(performance.get("bootstrap_samples") == BOOTSTRAP_SAMPLES, "bootstrap_samples must be exactly 20000")
    require(performance.get("bootstrap_seed") == BOOTSTRAP_SEED, f"bootstrap_seed must be {BOOTSTRAP_SEED}")
    workloads = performance.get("workloads")
    require(isinstance(workloads, list) and len(workloads) == len(WORKLOADS), "performance must contain exactly two workloads")
    require([workload.get("workload_id") for workload in workloads if isinstance(workload, Mapping)] == list(WORKLOADS), "workload order or identity differs from the sealed contract")
    block_ratios = {workload["workload_id"]: _workload_block_ratios(workload) for workload in workloads}
    workload_ratios = {name: _geomean(ratios) for name, ratios in block_ratios.items()}
    aggregate_geomean = _geomean(list(workload_ratios.values()))

    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    workload_samples = {name: [] for name in WORKLOADS}
    for _ in range(BOOTSTRAP_SAMPLES):
        replicate_workloads = []
        for name in WORKLOADS:
            ratios = block_ratios[name]
            replicate = _geomean([ratios[rng.randrange(len(ratios))] for _ in ratios])
            workload_samples[name].append(replicate)
            replicate_workloads.append(replicate)
        samples.append(_geomean(replicate_workloads))
    samples.sort()
    lower_95 = _percentile(samples, 0.025)
    upper_95 = _percentile(samples, 0.975)
    workload_intervals = {}
    for name, values in workload_samples.items():
        values.sort()
        workload_intervals[name] = {
            "lower_95": _percentile(values, 0.025),
            "upper_95": _percentile(values, 0.975),
        }

    require(aggregate_geomean > 1.0, f"aggregate throughput geomean {aggregate_geomean:.9f} is not > 1")
    require(lower_95 > 1.0, f"aggregate throughput lower 95% CI {lower_95:.9f} is not > 1")
    for name, interval in workload_intervals.items():
        require(interval["upper_95"] >= 1.0, f"workload {name} has a resolved regression (upper 95% CI {interval['upper_95']:.9f} < 1)")
    return {
        "workload_ratios": workload_ratios,
        "workload_intervals": workload_intervals,
        "aggregate_geomean": aggregate_geomean,
        "aggregate_lower_95": lower_95,
        "aggregate_upper_95": upper_95,
    }


def validate_campaign(campaign: Mapping[str, object]) -> dict[str, float]:
    physical = _finite_positive(campaign.get("physical_turnaround_seconds"), "physical turnaround")
    measured = _finite_positive(campaign.get("measured_runtime_seconds"), "measured campaign runtime")
    require(isinstance(campaign.get("started_at"), str) and campaign["started_at"], "campaign started_at is required")
    require(isinstance(campaign.get("finished_at"), str) and campaign["finished_at"], "campaign finished_at is required")
    require(physical >= measured, "physical turnaround cannot be shorter than measured runtime")
    return {"physical_turnaround_seconds": physical, "measured_runtime_seconds": measured}


def audit_receipt(receipt: Mapping[str, object], evidence_root: Path | None = None) -> dict[str, object]:
    require(receipt.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    for section in (
        "provenance",
        "campaign",
        "accuracy",
        "mtp_probe",
        "routes",
        "performance",
    ):
        require(isinstance(receipt.get(section), Mapping), f"{section} must be an object")
    validate_provenance(receipt["provenance"])
    campaign = validate_campaign(receipt["campaign"])
    accuracy = validate_accuracy(receipt["accuracy"], evidence_root)
    mtp_probe = validate_mtp_probe(receipt["mtp_probe"])
    routes = validate_routes(receipt["routes"])
    performance = validate_performance(receipt["performance"])
    return {
        "schema": "gdn-public-qualification-audit-v2",
        "passed": True,
        "campaign": campaign,
        "accuracy": accuracy,
        "mtp_probe": mtp_probe,
        "routes": routes,
        "performance": performance,
    }
