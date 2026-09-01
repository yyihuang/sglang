"""Collect source-owned raw observations from an already running SGLang server.

This client deliberately does not start or stop a server.  The rendered plan
defines the two server configurations and ABBA order; this module gives each
measurement a small, auditable, one-purpose command.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from tools.gdn_public_qualification.contract import (
    ACCURACY_SAMPLING_PARAMS,
    GSM8K_SHOTS,
    HASHES,
    KL_DISTRIBUTION_SCHEMA,
    KL_NORMALIZATION_ATOL,
    KL_SAMPLE_COUNT,
    KL_TOKEN_ID_ORDER,
    KL_VOCAB_CHUNK_SIZE,
    MTP_PROBE_MAX_NEW_TOKENS,
    MTP_PROBE_PROMPT_INDEX,
    MTP_SPECULATIVE_EAGLE_TOPK,
    MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
    MTP_SPECULATIVE_NUM_STEPS,
    PROMPT_COUNT,
    PLAN_SCHEMA,
    TP_SIZE,
    expected_server_config,
    validate_provenance,
)
from tools.gdn_public_qualification.kl_sink_hook import (
    KL_SINK_AUTHORITY_SCHEMA,
    KL_SINK_RESPONSE_KEY,
    KL_SINK_SAMPLE_SCHEMA,
    marker_for_sample,
)

INVALID_ANSWER = object()
ROUTE_MARKER = "FLASHINFER_GDN_NONCP_ROUTE"
ROUTE_RE = re.compile(
    r"FLASHINFER_GDN_NONCP_ROUTE\s+backend=gdn_noncp\s+"
    r"route=(flashinfer\.gdn_(prefill|decode)\.noncp\.[a-z0-9_.-]+)\s+"
    r"phase=(prefill|decode)\s+t=([1-9][0-9]*)\s+gates_present=True(?:\s|$)"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path} SHA256 {actual} != {expected}")


def _post(base_url: str, endpoint: str, payload: dict, timeout: float) -> object:
    response = requests.post(base_url.rstrip("/") + endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get(base_url: str, endpoint: str, timeout: float) -> object:
    response = requests.get(base_url.rstrip("/") + endpoint, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _flush(base_url: str, timeout: float) -> None:
    response = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=timeout)
    response.raise_for_status()


def _answer_value(text: str) -> object:
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    try:
        value = ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID_ANSWER
    return value if isinstance(value, (int, float)) and math.isfinite(value) else INVALID_ANSWER


def collect_accuracy(args: argparse.Namespace) -> None:
    if args.dataset_sha256 != HASHES["gsm8k_dataset_sha256"]:
        raise ValueError("GSM8K dataset hash differs from the sealed qualification authority")
    if args.output.exists():
        raise ValueError(f"accuracy output already exists: {args.output}")
    if args.ledger.exists():
        raise ValueError(f"accuracy request ledger already exists: {args.ledger}")
    if args.output.resolve().parent != args.ledger.resolve().parent:
        raise ValueError("accuracy output and request ledger must share one evidence directory")
    if _sha256(args.plan) != args.campaign_id:
        raise ValueError("accuracy campaign_id differs from the rendered plan SHA256")
    try:
        plan = json.loads(args.plan.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"rendered qualification plan is unreadable: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("accuracy plan schema differs from the qualification contract")
    if not isinstance(plan.get("provenance"), dict):
        raise ValueError("accuracy plan provenance is required")
    validate_provenance(plan["provenance"])
    plan_accuracy = plan.get("accuracy")
    if not isinstance(plan_accuracy, dict) or plan_accuracy.get("prompt_count") != PROMPT_COUNT:
        raise ValueError("accuracy plan does not bind all sealed prompts")
    plan_servers = plan.get("servers")
    if (
        not isinstance(plan_servers, dict)
        or not isinstance(plan_servers.get(args.arm), dict)
        or plan_servers[args.arm].get("base_url") != args.base_url.rstrip("/")
    ):
        raise ValueError(f"accuracy {args.arm} base URL differs from the rendered plan")
    _require_hash(args.dataset, args.dataset_sha256)
    if args.prompt_ids_sha256 != HASHES["gsm8k_prompt_ids_sha256"]:
        raise ValueError("GSM8K prompt IDs hash differs from the sealed qualification authority")
    _require_hash(args.prompt_ids, args.prompt_ids_sha256)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if len(rows) != GSM8K_SHOTS + PROMPT_COUNT:
        raise ValueError(f"sealed GSM8K file must contain exactly 1319 rows, got {len(rows)}")
    prompt_ids = _load_input_ids(args.prompt_ids, PROMPT_COUNT)
    if not re.fullmatch(r"[0-9a-f]{64}", args.campaign_id):
        raise ValueError("accuracy campaign_id must be a SHA256")
    if args.model_manifest_sha256 != HASHES["model_manifest_sha256"]:
        raise ValueError("accuracy model manifest hash differs from the sealed authority")
    server_info = _get(args.base_url, "/server_info", args.timeout)
    model_info = _get(args.base_url, "/model_info", args.timeout)
    if not isinstance(server_info, dict) or not isinstance(model_info, dict):
        raise ValueError("accuracy server identity endpoints must return objects")
    expected_config = expected_server_config(args.arm)
    server_config = {key: server_info.get(key) for key in expected_config}
    if server_config != expected_config:
        raise ValueError(
            f"accuracy {args.arm} server configuration {server_config!r} != {expected_config!r}"
        )
    model_identity = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "model_manifest_sha256": args.model_manifest_sha256,
    }
    observed_model_identity = {
        "model_path": model_info.get("model_path"),
        "tokenizer_path": model_info.get("tokenizer_path"),
        "model_manifest_sha256": args.model_manifest_sha256,
    }
    if observed_model_identity != model_identity:
        raise ValueError(
            f"accuracy {args.arm} server model identity {observed_model_identity!r} "
            f"!= {model_identity!r}"
        )

    ledger = args.ledger.open("x")
    ledger_lock = threading.Lock()

    def record_event(value: dict) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with ledger_lock:
            ledger.write(encoded)
            ledger.flush()

    record_event(
        {
            "schema": "gdn-gsm8k-request-ledger-v1",
            "arm": args.arm,
            "campaign_id": args.campaign_id,
            "prompt_count": PROMPT_COUNT,
        }
    )

    def run(question_index: int) -> dict:
        source_row_index = GSM8K_SHOTS + question_index
        input_ids = prompt_ids[question_index]
        request_id = f"gdn-gsm8k-{args.campaign_id}-{args.arm}-{question_index:04d}"
        payload = {
            "rid": request_id,
            "input_ids": input_ids,
            "sampling_params": ACCURACY_SAMPLING_PARAMS,
        }
        record_event(
            {
                "event": "dispatch",
                "question_index": question_index,
                "request_id": request_id,
                "payload_sha256": _json_sha256(payload),
            }
        )
        result = _post(args.base_url, "/generate", payload, args.timeout)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("text"), str)
            or not isinstance(result.get("output_ids"), list)
            or not result["output_ids"]
            or not all(type(token) is int for token in result["output_ids"])
            or not isinstance(result.get("meta_info"), dict)
            or result["meta_info"].get("id") != request_id
        ):
            raise ValueError(f"prompt {question_index} returned an invalid response")
        expected = _answer_value(rows[source_row_index]["answer"])
        observed = _answer_value(result["text"])
        record_event(
            {
                "event": "response",
                "question_index": question_index,
                "request_id": request_id,
                "response_sha256": _json_sha256(result),
            }
        )
        return {
            "question_index": question_index,
            "source_row_index": source_row_index,
            "request_count": 1,
            "request_id": request_id,
            "input_ids_sha256": _json_sha256(input_ids),
            "input_token_count": len(input_ids),
            "output_ids_sha256": _json_sha256(result["output_ids"]),
            "response": result,
            "correct": observed is not INVALID_ANSWER and observed == expected,
        }

    try:
        _flush(args.base_url, args.timeout)
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            prompt_rows = list(executor.map(run, range(PROMPT_COUNT)))
        ledger.flush()
        os.fsync(ledger.fileno())
    finally:
        ledger.close()
    score = sum(row["correct"] for row in prompt_rows) / PROMPT_COUNT
    _write_json_exclusive(
        args.output,
        {
            "arm": args.arm,
            "campaign_id": args.campaign_id,
            "plan_sha256": args.campaign_id,
            "dataset_sha256": args.dataset_sha256,
            "prompt_ids_sha256": args.prompt_ids_sha256,
            "request_payload": "input_ids",
            "sampling_params": ACCURACY_SAMPLING_PARAMS,
            "server_config": server_config,
            "model_identity": model_identity,
            "request_ledger": {
                "path": args.ledger.name,
                "sha256": _sha256(args.ledger),
            },
            "score": score,
            "prompts": prompt_rows,
        },
    )


def _load_input_ids(path: Path, expected_count: int) -> list[list[int]]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key in ("input_ids", "prompts", "records"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = [row["input_ids"] for row in value]
    if not (
        isinstance(value, list)
        and len(value) == expected_count
        and all(isinstance(row, list) and row and all(isinstance(token, int) for token in row) for row in value)
    ):
        raise ValueError(f"{path} must contain exactly {expected_count} non-empty token-ID rows")
    return value


def _as_results(value: object, expected: int) -> list[dict]:
    rows = value if isinstance(value, list) else [value]
    if len(rows) != expected or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"server returned {len(rows)} results, expected {expected}")
    return rows


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ValueError("NumPy from the pinned SGLang runtime is required for full-vocabulary KL collection") from exc
    return np


def _kl_server_identity(args: argparse.Namespace) -> dict[str, object]:
    if args.model_manifest_sha256 != HASHES["model_manifest_sha256"]:
        raise ValueError("KL model manifest hash differs from the sealed qualification authority")
    if args.input_ids_sha256 != HASHES["longbench_first48_ids_sha256"]:
        raise ValueError("KL input IDs hash differs from the sealed 48-sample authority")
    if type(args.vocab_size) is not int or args.vocab_size <= 1:
        raise ValueError("KL vocab_size must be an integer > 1")
    if Path(args.tokenizer_path).resolve() != Path(args.model_path).resolve():
        raise ValueError("KL tokenizer path must be the tokenizer inside the sealed model directory")
    config_path = Path(args.model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"KL model config is missing: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"KL model config must be an object: {config_path}")
    configured_vocab_size = config.get("vocab_size")
    if configured_vocab_size is None and isinstance(config.get("text_config"), dict):
        configured_vocab_size = config["text_config"].get("vocab_size")
    if configured_vocab_size != args.vocab_size:
        raise ValueError(
            f"KL vocab_size {args.vocab_size} != sealed model config value "
            f"{configured_vocab_size!r}"
        )
    server_info = _get(args.base_url, "/server_info", args.timeout)
    model_info = _get(args.base_url, "/model_info", args.timeout)
    if not isinstance(server_info, dict) or not isinstance(model_info, dict):
        raise ValueError("KL server identity endpoints must return objects")
    if server_info.get("tp_size") != TP_SIZE:
        raise ValueError(f"KL server tp_size {server_info.get('tp_size')!r} != {TP_SIZE}")
    expected = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
    }
    observed = {key: model_info.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"KL server model/tokenizer identity {observed!r} != {expected!r}")
    return {
        **expected,
        "model_manifest_sha256": args.model_manifest_sha256,
        "vocab_size": args.vocab_size,
    }


def _prepare_kl_output(args: argparse.Namespace, arm: str) -> tuple[dict, str]:
    root = args.sink_root.resolve()
    if args.output.resolve() != root / "manifest.json":
        raise ValueError("KL output must be the sealed sink root manifest.json")
    if args.output.exists():
        raise ValueError(f"KL manifest output already exists: {args.output}")
    authority_path = root / "authority.json"
    if not authority_path.is_file():
        raise ValueError("KL sink authority is missing")
    raw = authority_path.read_bytes()
    authority = json.loads(raw)
    expected = {
        "schema": KL_SINK_AUTHORITY_SCHEMA,
        "root": str(root),
        "arm": arm,
        "vocab_size": args.vocab_size,
        "sample_count": KL_SAMPLE_COUNT,
        "position_count": 512,
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "normalization_atol": KL_NORMALIZATION_ATOL,
    }
    if not isinstance(authority, dict) or any(authority.get(k) != v for k, v in expected.items()):
        raise ValueError("KL sink authority differs from the collector contract")
    for directory in (root / "shards", root / "receipts"):
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(f"KL sink directory must be fresh and empty: {directory}")
    return authority, hashlib.sha256(raw).hexdigest()


def _extract_sink_receipt(
    result: object,
    output_ids: list[int],
    sample_index: int,
    authority_sha256: str,
    arm: str,
    vocab_size: int,
) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("meta_info"), dict):
        raise ValueError(f"KL sample {sample_index} response has no meta_info")
    meta_info = result["meta_info"]
    selected = meta_info.get("input_token_logprobs")
    position_count = len(output_ids)
    if not isinstance(selected, list) or len(selected) < position_count:
        raise ValueError(f"KL sample {sample_index} returned incomplete teacher-forced positions")
    selected = selected[-position_count:]
    for position, (selected_row, expected_output_id) in enumerate(zip(selected, output_ids)):
        if not (
            isinstance(selected_row, (list, tuple))
            and len(selected_row) >= 2
            and selected_row[1] == expected_output_id
        ):
            raise ValueError(f"KL sample {sample_index} position {position} teacher-forced token alignment differs")
    response = meta_info.get(KL_SINK_RESPONSE_KEY)
    if not isinstance(response, dict) or not isinstance(response.get("receipt"), dict):
        raise ValueError(f"KL sample {sample_index} returned no sealed sink receipt")
    receipt = response["receipt"]
    expected = {
        "schema": KL_SINK_SAMPLE_SCHEMA,
        "authority_sha256": authority_sha256,
        "arm": arm,
        "sample_index": sample_index,
        "position_count": position_count,
        "vocab_size": vocab_size,
        "position_mapping": "first_512_rows_after_prompt_predecessor",
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
    }
    if any(receipt.get(k) != v for k, v in expected.items()) or not isinstance(
        response.get("receipt_sha256"), str
    ):
        raise ValueError(f"KL sample {sample_index} sink receipt identity differs")
    return response


def _collect_kl_distribution(
    args: argparse.Namespace,
    arm: str,
    identity: dict[str, object],
    input_ids: list[list[int]],
    continuations: list[list[int]],
    reference_manifest_sha256: str | None = None,
) -> None:
    """Request one server-side full-vocabulary sink forward per sample."""

    np = _require_numpy()
    _, authority_sha256 = _prepare_kl_output(args, arm)
    root = args.sink_root.resolve()
    records = []
    collection_start = time.perf_counter()
    _flush(args.base_url, args.timeout)
    for sample_index, (prompt_ids, output_ids) in enumerate(zip(input_ids, continuations)):
        position_count = len(output_ids)
        probability_mass = np.zeros(position_count, dtype=np.float64)
        joined = prompt_ids + output_ids
        result = _as_results(
            _post(
                args.base_url,
                "/generate",
                {
                    "input_ids": [joined],
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
                    "return_logprob": True,
                    "return_text_in_logprobs": False,
                    "logprob_start_len": max(0, len(prompt_ids) - 1),
                    "token_ids_logprob": marker_for_sample(args.vocab_size, sample_index),
                },
                args.timeout,
            ),
            1,
        )[0]
        response = _extract_sink_receipt(
            result, output_ids, sample_index, authority_sha256, arm, args.vocab_size
        )
        receipt_path = root / "receipts" / f"sample-{sample_index:03d}.json"
        raw_receipt = receipt_path.read_bytes()
        if hashlib.sha256(raw_receipt).hexdigest() != response["receipt_sha256"]:
            raise ValueError(f"KL sample {sample_index} receipt SHA256 mismatch")
        receipt = json.loads(raw_receipt)
        if receipt != response["receipt"] or receipt.get("arm") != arm:
            raise ValueError(f"KL sample {sample_index} receipt content differs")
        shards = receipt.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"KL sample {sample_index} has no sink shards")
        cursor = 0
        for shard_index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError(f"KL sample {sample_index} shard {shard_index} is malformed")
            token_start, token_end = shard.get("token_start"), shard.get("token_end")
            if (
                type(token_start) is not int
                or type(token_end) is not int
                or token_start != cursor
                or not token_start < token_end <= args.vocab_size
                or token_end - token_start > KL_VOCAB_CHUNK_SIZE
                or shard.get("shape") != [position_count, token_end - token_start]
                or shard.get("byte_count") != position_count * (token_end - token_start) * 4
            ):
                raise ValueError(f"KL sample {sample_index} shard {shard_index} coverage differs")
            expected_name = (
                f"sample-{sample_index:03d}-vocab-{token_start:06d}-{token_end:06d}.f32le"
            )
            if shard.get("path") != f"shards/{expected_name}":
                raise ValueError(f"KL sample {sample_index} shard {shard_index} path differs")
            path = root / "shards" / expected_name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"KL sample {sample_index} shard {shard_index} is missing")
            raw = path.read_bytes()
            if len(raw) != shard.get("byte_count") or hashlib.sha256(raw).hexdigest() != shard.get("sha256"):
                raise ValueError(f"KL sample {sample_index} shard {shard_index} hash or size differs")
            matrix = np.frombuffer(raw, dtype="<f4").reshape(position_count, token_end - token_start).astype(np.float64)
            if not bool(np.isfinite(matrix).all()) or float(np.max(matrix)) > 1e-6:
                raise ValueError(f"KL sample {sample_index} shard {shard_index} contains invalid logprobs")
            probability_mass += np.sum(np.exp(matrix.astype(np.float64)), axis=1, dtype=np.float64)
            cursor = token_end
        if cursor != args.vocab_size:
            raise ValueError(f"KL sample {sample_index} does not cover the full vocabulary")
        bad_positions = np.flatnonzero(np.abs(probability_mass - 1.0) > KL_NORMALIZATION_ATOL)
        if bad_positions.size:
            position = int(bad_positions[0])
            raise ValueError(
                f"KL sample {sample_index} position {position} probability mass "
                f"{float(probability_mass[position]):.9f} is not normalized"
            )
        records.append(
            {
                "sample_index": sample_index,
                "input_ids_sha256": _json_sha256(prompt_ids),
                "output_ids": output_ids,
                "output_ids_sha256": _json_sha256(output_ids),
                "position_count": position_count,
                "shards": shards,
            }
        )
    manifest = {
        "schema": KL_DISTRIBUTION_SCHEMA,
        "arm": arm,
        "sample_count": KL_SAMPLE_COUNT,
        "input_ids_sha256": args.input_ids_sha256,
        **identity,
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "normalization_atol": KL_NORMALIZATION_ATOL,
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
        "sink_authority_sha256": authority_sha256,
        "collection_runtime_seconds": time.perf_counter() - collection_start,
        "records": records,
    }
    if reference_manifest_sha256 is not None:
        manifest["reference_manifest_sha256"] = reference_manifest_sha256
    _write_json(args.output, manifest)


def collect_mtp_probe(args: argparse.Namespace) -> None:
    """Run one real T=4 NEXTN request against an already loaded TP4 server."""

    _require_hash(args.input_ids, args.input_ids_sha256)
    input_ids = _load_input_ids(args.input_ids, KL_SAMPLE_COUNT)
    server_info = _get(args.base_url, "/server_info", args.timeout)
    if not isinstance(server_info, dict):
        raise ValueError("MTP probe server_info response must be an object")
    expected_config = expected_server_config(args.arm)
    server_config = {key: server_info.get(key) for key in expected_config}
    if server_config != expected_config:
        raise ValueError(
            f"MTP probe {args.arm} server configuration {server_config!r} "
            f"!= {expected_config!r}"
        )

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": MTP_PROBE_MAX_NEW_TOKENS,
        "ignore_eos": True,
    }
    _flush(args.base_url, args.timeout)
    start = time.perf_counter()
    result = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": [input_ids[MTP_PROBE_PROMPT_INDEX]],
                "sampling_params": sampling_params,
            },
            args.timeout,
        ),
        1,
    )[0]
    elapsed = time.perf_counter() - start
    output_ids = result.get("output_ids")
    if not (
        isinstance(output_ids, list)
        and len(output_ids) == MTP_PROBE_MAX_NEW_TOKENS
        and all(type(token) is int for token in output_ids)
    ):
        raise ValueError(
            "MTP probe must return exactly "
            f"{MTP_PROBE_MAX_NEW_TOKENS} integer output IDs"
        )
    _write_json(
        args.output,
        {
            "arm": args.arm,
            "input_ids_sha256": args.input_ids_sha256,
            "prompt_index": MTP_PROBE_PROMPT_INDEX,
            "request_count": 1,
            "sampling_params": sampling_params,
            "server_config": server_config,
            "output_ids": output_ids,
            "output_ids_sha256": _json_sha256(output_ids),
            "measured_runtime_seconds": elapsed,
        },
    )


def collect_kl_reference(args: argparse.Namespace) -> None:
    """Generate sealed continuations, then score their full baseline distributions."""

    _require_hash(args.input_ids, args.input_ids_sha256)
    input_ids = _load_input_ids(args.input_ids, KL_SAMPLE_COUNT)
    identity = _kl_server_identity(args)
    _flush(args.base_url, args.timeout)
    results = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 512, "ignore_eos": True},
                "return_logprob": True,
                "return_text_in_logprobs": False,
            },
            args.timeout,
        ),
        KL_SAMPLE_COUNT,
    )
    continuations = []
    for sample_index, result in enumerate(results):
        output_ids = result.get("output_ids")
        if not (
            isinstance(output_ids, list)
            and len(output_ids) == 512
            and all(type(token) is int and 0 <= token < args.vocab_size for token in output_ids)
        ):
            raise ValueError(f"KL reference sample {sample_index} must return exactly 512 in-vocabulary token IDs")
        continuations.append(output_ids)
    _collect_kl_distribution(args, "baseline", identity, input_ids, continuations)


def collect_kl_candidate(args: argparse.Namespace) -> None:
    reference_sha256 = _sha256(args.reference)
    reference = json.loads(args.reference.read_text())
    if not isinstance(reference, dict) or reference.get("schema") != KL_DISTRIBUTION_SCHEMA:
        raise ValueError("KL reference must be a full-vocabulary distribution manifest")
    if reference.get("arm") != "baseline" or reference.get("sample_count") != KL_SAMPLE_COUNT:
        raise ValueError("KL reference must contain exactly 48 baseline samples")
    identity = _kl_server_identity(args)
    for key, expected in {
        "input_ids_sha256": HASHES["longbench_first48_ids_sha256"],
        "model_manifest_sha256": args.model_manifest_sha256,
        "model_path": identity["model_path"],
        "tokenizer_path": identity["tokenizer_path"],
        "vocab_size": args.vocab_size,
        "token_id_order": KL_TOKEN_ID_ORDER,
    }.items():
        if reference.get(key) != expected:
            raise ValueError(f"KL reference {key} differs from the candidate server authority")
    records = reference.get("records")
    if not isinstance(records, list) or len(records) != KL_SAMPLE_COUNT:
        raise ValueError("KL reference must contain exactly 48 ordered records")
    input_ids = _load_input_ids(args.input_ids, KL_SAMPLE_COUNT)
    _require_hash(args.input_ids, args.input_ids_sha256)
    if args.input_ids_sha256 != reference.get("input_ids_sha256"):
        raise ValueError("candidate input-ID artifact differs from the baseline manifest")
    continuations = []
    for sample_index, (tokens, row) in enumerate(zip(input_ids, records)):
        if not isinstance(row, dict) or row.get("sample_index") != sample_index:
            raise ValueError(f"KL reference sample {sample_index} identity differs")
        if row.get("input_ids_sha256") != _json_sha256(tokens):
            raise ValueError(f"KL reference sample {sample_index} input IDs differ")
        output_ids = row.get("output_ids")
        if not (
            isinstance(output_ids, list)
            and len(output_ids) == 512
            and all(type(token) is int and 0 <= token < args.vocab_size for token in output_ids)
            and row.get("output_ids_sha256") == _json_sha256(output_ids)
        ):
            raise ValueError(f"KL reference sample {sample_index} output IDs are invalid")
        continuations.append(output_ids)
    _collect_kl_distribution(
        args,
        "candidate",
        identity,
        input_ids,
        continuations,
        reference_manifest_sha256=reference_sha256,
    )


def collect_performance(args: argparse.Namespace) -> None:
    _require_hash(args.input_ids, args.input_ids_sha256)
    input_ids = _load_input_ids(args.input_ids, args.prompt_count)
    _flush(args.base_url, args.timeout)
    start = time.perf_counter()
    results = _as_results(
        _post(
            args.base_url,
            "/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 512, "ignore_eos": True},
            },
            args.timeout,
        ),
        args.prompt_count,
    )
    elapsed = time.perf_counter() - start
    output_tokens = sum(len(row.get("output_ids", [])) for row in results)
    if output_tokens != args.prompt_count * 512:
        raise ValueError(f"performance run returned {output_tokens} output tokens, expected {args.prompt_count * 512}")
    _write_json(
        args.output,
        {
            "arm": args.arm,
            "workload_id": args.workload_id,
            "input_ids_sha256": args.input_ids_sha256,
            "throughput_tokens_per_second": output_tokens / elapsed,
            "measured_runtime_seconds": elapsed,
            "output_tokens": output_tokens,
        },
    )


def collect_routes(args: argparse.Namespace) -> None:
    rank_rows = []
    for rank_log in args.rank_log:
        rank_text, path_text = rank_log.split(":", 1)
        rank = int(rank_text)
        path = Path(path_text)
        text = path.read_text(errors="replace")
        marker_lines = [line for line in text.splitlines() if ROUTE_MARKER in line]
        observations = set()
        for line in marker_lines:
            match = ROUTE_RE.search(line)
            if match is None:
                raise ValueError(f"malformed optimized GDN route marker in TP{rank}: {line!r}")
            route, route_phase, phase, token_width = match.groups()
            if route_phase != phase:
                raise ValueError(f"route/phase mismatch in TP{rank}: {line!r}")
            observations.add((route, phase, int(token_width)))
        prefill = sorted({route for route, phase, _ in observations if phase == "prefill"})
        decode = sorted({route for route, phase, _ in observations if phase == "decode"})
        rank_rows.append(
            {
                "rank": rank,
                "prefill_routes": prefill,
                "decode_routes": decode,
                "route_observations": [
                    {"route": route, "phase": phase, "t": token_width, "gates_present": True}
                    for route, phase, token_width in sorted(observations)
                ],
                "marker_count": len(marker_lines),
                "fallback_count": len(re.findall(r"fall(?:ing)? back", text, flags=re.IGNORECASE)),
                "route_error_count": len(re.findall(r"(?:FLASHINFER_GDN_NONCP_ROUTE|backend=gdn_noncp).*?(?:error|failed)", text, flags=re.IGNORECASE)),
                "log_sha256": _sha256(path),
            }
        )
    rank_rows.sort(key=lambda row: row["rank"])
    _write_json(args.output, {"arm": args.arm, "ranks": rank_rows})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def server_parser(name: str, function):
        child = subparsers.add_parser(name)
        child.add_argument("--base-url", required=True)
        child.add_argument("--timeout", type=float, default=7200)
        child.add_argument("--output", type=Path, required=True)
        child.set_defaults(function=function)
        return child

    accuracy = server_parser("accuracy", collect_accuracy)
    accuracy.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    accuracy.add_argument("--campaign-id", required=True, help="SHA256 of the sealed campaign plan")
    accuracy.add_argument("--plan", type=Path, required=True)
    accuracy.add_argument("--ledger", type=Path, required=True)
    accuracy.add_argument("--dataset", type=Path, required=True)
    accuracy.add_argument("--dataset-sha256", required=True)
    accuracy.add_argument("--prompt-ids", type=Path, required=True)
    accuracy.add_argument("--prompt-ids-sha256", required=True)
    accuracy.add_argument("--model-path", required=True)
    accuracy.add_argument("--tokenizer-path", required=True)
    accuracy.add_argument("--model-manifest-sha256", required=True)
    accuracy.add_argument("--parallel", type=int, default=128)

    kl_reference = server_parser("kl-reference", collect_kl_reference)
    kl_reference.add_argument("--input-ids", type=Path, required=True)
    kl_reference.add_argument("--input-ids-sha256", required=True)

    kl_candidate = server_parser("kl-candidate", collect_kl_candidate)
    kl_candidate.add_argument("--reference", type=Path, required=True)
    kl_candidate.add_argument("--input-ids", type=Path, required=True)
    kl_candidate.add_argument("--input-ids-sha256", required=True)

    for child in (kl_reference, kl_candidate):
        child.add_argument("--model-path", required=True)
        child.add_argument("--tokenizer-path", required=True)
        child.add_argument("--model-manifest-sha256", required=True)
        child.add_argument("--vocab-size", type=int, required=True)
        child.add_argument("--sink-root", type=Path, required=True)

    mtp_probe = server_parser("mtp-probe", collect_mtp_probe)
    mtp_probe.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    mtp_probe.add_argument("--input-ids", type=Path, required=True)
    mtp_probe.add_argument("--input-ids-sha256", required=True)

    performance = server_parser("performance", collect_performance)
    performance.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    performance.add_argument("--workload-id", required=True)
    performance.add_argument("--input-ids", type=Path, required=True)
    performance.add_argument("--input-ids-sha256", required=True)
    performance.add_argument("--prompt-count", type=int, choices=(32, 48), required=True)

    routes = subparsers.add_parser("routes")
    routes.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    routes.add_argument("--rank-log", action="append", required=True, help="RANK:PATH; pass exactly four")
    routes.add_argument("--output", type=Path, required=True)
    routes.set_defaults(function=collect_routes)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
