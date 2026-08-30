"""Immutable inputs and gates for the public GDN qualification campaign.

The campaign runner writes raw observations.  ``audit.py`` independently
recomputes every acceptance metric from those observations and refuses partial,
duplicated, or provenance-drifted evidence.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence

SCHEMA = "gdn-public-qualification-result-v1"
PLAN_SCHEMA = "gdn-public-qualification-plan-v1"

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
TP_SIZE = 4
TP_RANKS = [0, 1, 2, 3]
ABBA_ORDER = ["baseline", "candidate", "candidate", "baseline"] * 4
OBSERVATIONS_PER_ARM_PER_WORKLOAD = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 2026083001

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ROUTE_RE = {
    "prefill": re.compile(r"^cake\.gdn_prefill\.noncp\.[a-z0-9_.-]+$"),
    "decode": re.compile(r"^cake\.gdn_decode\.noncp\.[a-z0-9_.-]+$"),
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


def _recompute_kl(kl: Mapping[str, object]) -> float:
    rows = kl.get("records")
    require(isinstance(rows, list) and len(rows) == KL_SAMPLE_COUNT, "KL must contain exactly 48 sealed samples")
    require([row.get("sample_index") for row in rows if isinstance(row, Mapping)] == list(range(KL_SAMPLE_COUNT)), "KL sample indices must be exactly 0..47 in order")
    sample_means = []
    for sample_index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"KL sample {sample_index} must be an object")
        baseline = row.get("baseline_logprobs")
        candidate = row.get("candidate_logprobs")
        require(isinstance(baseline, list) and isinstance(candidate, list), f"KL sample {sample_index} logprobs must be lists")
        require(len(baseline) == len(candidate) and len(baseline) > 0, f"KL sample {sample_index} logprob lengths differ or are empty")
        terms = []
        for token_index, (baseline_value, candidate_value) in enumerate(zip(baseline, candidate)):
            require(isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)), f"KL sample {sample_index} token {token_index} is not numeric")
            baseline_value = float(baseline_value)
            candidate_value = float(candidate_value)
            require(math.isfinite(baseline_value) and math.isfinite(candidate_value), f"KL sample {sample_index} token {token_index} is not finite")
            logr = baseline_value - candidate_value
            try:
                term = math.exp(logr) - 1.0 - logr
            except OverflowError as exc:
                raise QualificationError(f"KL sample {sample_index} overflowed") from exc
            require(math.isfinite(term), f"KL sample {sample_index} term is not finite")
            terms.append(term)
        sample_means.append(sum(terms) / len(terms))
    return sum(sample_means) / len(sample_means)


def validate_accuracy(accuracy: Mapping[str, object]) -> dict[str, float]:
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
    mean_kl = _recompute_kl(kl)
    reported_kl = kl.get("mean_kl")
    require(isinstance(reported_kl, (int, float)) and math.isclose(float(reported_kl), mean_kl, rel_tol=0, abs_tol=1e-15), "reported KL does not match raw logprobs")
    require(mean_kl < MAX_KL_EXCLUSIVE, f"mean KL {mean_kl:.9f} is not < 0.0035")
    return {**scores, "mean_kl": mean_kl}


def _validate_route_names(route_kind: str, routes: object) -> list[str]:
    require(isinstance(routes, list) and bool(routes), f"candidate {route_kind} routes must be a non-empty list")
    require(routes == sorted(set(routes)), f"candidate {route_kind} routes must be sorted and unique")
    for route in routes:
        require(isinstance(route, str) and bool(_ROUTE_RE[route_kind].fullmatch(route)), f"invalid {route_kind} non-CP route {route!r}")
    return routes


def validate_routes(routes: Mapping[str, object]) -> dict[str, list[str]]:
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
            if arm == "candidate":
                require(prefill == expected_routes["prefill"], f"candidate rank {row.get('rank')} prefill routes differ")
                require(decode == expected_routes["decode"], f"candidate rank {row.get('rank')} decode routes differ")
                require(row.get("cake_route_count") == len(prefill) + len(decode), f"candidate rank {row.get('rank')} route count differs")
            else:
                require(prefill == [] and decode == [], f"baseline rank {row.get('rank')} recorded Cake routes")
                require(row.get("cake_route_count") == 0, f"baseline rank {row.get('rank')} Cake route count is nonzero")
    return expected_routes


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


def audit_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    require(receipt.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    for section in ("provenance", "campaign", "accuracy", "routes", "performance"):
        require(isinstance(receipt.get(section), Mapping), f"{section} must be an object")
    validate_provenance(receipt["provenance"])
    campaign = validate_campaign(receipt["campaign"])
    accuracy = validate_accuracy(receipt["accuracy"])
    routes = validate_routes(receipt["routes"])
    performance = validate_performance(receipt["performance"])
    return {
        "schema": "gdn-public-qualification-audit-v1",
        "passed": True,
        "campaign": campaign,
        "accuracy": accuracy,
        "routes": routes,
        "performance": performance,
    }
