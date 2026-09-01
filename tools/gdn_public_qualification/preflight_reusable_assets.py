#!/usr/bin/env python3
"""Revalidate sealed, winner-independent assets before a GDN campaign launch.

This command deliberately does not launch a model server, inspect candidate
routes, or claim final-campaign evidence.  It hashes the immutable model,
prompt/workload, and runtime-dependency assets that may be reused after the
final kernel winner is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "gdn-public-qualification-reusable-assets-preflight-v1"
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_REVISION = "c5f5f263bdd5cc134092897864e8905d8fe7b928"
MODEL_MANIFEST_SHA256 = (
    "49f46f7e1b93abad35e295348cc8e1477b3df1a0c597a97791ac7d7a6d7b0a06"
)
MODEL_STAGE_RECEIPT_SHA256 = (
    "374dc629e32e2a1c5d50972fd38893a8f627d94e71450ef8c86784a42359f2b8"
)
MODEL_TRANSFER_RECEIPT_SHA256 = (
    "f542acc9b038f87359eb7d2df8067dea2ca4b45eff906902290214b4e10d7075"
)
MODEL_VERIFICATION_RESULT_SHA256 = (
    "dd02bbd379abf21770fb0ef654e080a71ec98f04fed70d48d276d62459cecda2"
)
MODEL_FILE_COUNT = 18
MODEL_BYTES = 82_082_296_496
MODEL_WEIGHT_FILE_COUNT = 8
MODEL_WEIGHT_BYTES = 82_051_854_384

INPUT_MANIFEST_SHA256 = (
    "f9b6aac29e058694edccfd6379e738db526c12e37a40f0773fe7e65af63024ef"
)
INPUT_FILES = {
    "gsm8k-manifest.json": "0b6bdeda8b61ffb2d25e83c78191ba94ed1ba295eaa558e8a72865a6bcc5a5a5",
    "gsm8k-prompt-token-ids.json": "6f2e88c9df2642a658293f7c0a7dd30ea8320414d2b9204b6130b10a06035e8a",
    "gsm8k-test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "longbench-first32-input-ids.json": "5f957f68a8f105b20c40fa5d49b8237c49b488b7b3298a8fb43535e41ace2033",
    "longbench-first48-input-ids.json": "36eb64beb3957576a28bdde7a51698912f01b8ae9c6a0353a8413a322aac47b7",
    "longbench-manifest.json": "c47e946834ae2dcff7aa7a372dbec0a55f6b94ae68308213f8d6d55f189a2342",
    "longbench-v2-data.json": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
    "source-authority.json": "14b3175be84dcc57c681300868ff97e2992a5da7184df1b002f233b9ffef1c09",
}

WHEEL_MANIFEST_SHA256 = (
    "7178c836e718d72dd456c5a5c2ce16f19c664dea9fc9dcb18f4670d8a91fdf0b"
)
WHEEL_FILENAMES = {
    "nvidia_cuda_nvdisasm-13.3.73-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
    "nvidia_cutlass_dsl-4.7.1-py3-none-any.whl",
    "nvidia_cutlass_dsl_libs_base-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl",
    "nvidia_cutlass_dsl_libs_core-4.7.1-py3-none-any.whl",
    "nvidia_cutlass_dsl_libs_cu12-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl",
    "nvidia_cutlass_dsl_libs_cu13-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


class PreflightError(ValueError):
    """A reusable asset differs from its sealed authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    _require(path.is_file() and not path.is_symlink(), f"missing regular JSON file: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PreflightError(f"strict JSON parse failed for {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: Path, expected_sha256: str) -> dict[str, object]:
    actual = _sha256(path)
    _require(actual == expected_sha256, f"SHA256 differs for {path}: {actual}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def _checksum_manifest(path: Path, root: Path) -> dict[str, str]:
    _require(path.is_file() and not path.is_symlink(), f"missing checksum manifest: {path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _CHECKSUM_RE.fullmatch(line)
        _require(match is not None, f"malformed checksum line {line_number} in {path}")
        digest, relative_text = match.groups()
        relative = Path(relative_text)
        _require(
            relative_text == relative.as_posix()
            and not relative.is_absolute()
            and relative_text not in ("", ".", "..")
            and ".." not in relative.parts,
            f"unsafe checksum path {relative_text!r} in {path}",
        )
        _require(relative_text not in entries, f"duplicate checksum path {relative_text!r}")
        target = root / relative
        _require(target.resolve().is_relative_to(root.resolve()), f"checksum path escapes root: {target}")
        _require(_sha256(target) == digest, f"checksum mismatch for {target}")
        entries[relative_text] = digest
    _require(bool(entries), f"empty checksum manifest: {path}")
    return entries


def _verify_inputs(input_root: Path) -> dict[str, object]:
    _require(input_root.is_absolute() and input_root.is_dir(), "input root must be an absolute directory")
    artifacts = input_root / "artifacts"
    _require(artifacts.is_dir() and not artifacts.is_symlink(), "sealed artifacts directory is missing")
    manifest = artifacts / "ARTIFACTS.sha256"
    manifest_record = _verified_file(manifest, INPUT_MANIFEST_SHA256)
    entries = _checksum_manifest(manifest, artifacts)
    _require(entries == INPUT_FILES, "sealed input artifact manifest entries differ")
    actual_names = {path.name for path in artifacts.iterdir()}
    _require(actual_names == {*INPUT_FILES, "ARTIFACTS.sha256"}, "sealed input artifact file set differs")

    gsm8k = _load_json(artifacts / "gsm8k-manifest.json")
    _require(isinstance(gsm8k, Mapping), "GSM8K manifest must be an object")
    _require(gsm8k.get("schema") == "gdn-gsm8k-1314-five-shot-manifest-v1", "GSM8K schema differs")
    _require(gsm8k.get("total_rows") == 1319, "GSM8K total row count differs")
    prompt_records = gsm8k.get("prompt_records")
    _require(isinstance(prompt_records, list) and len(prompt_records) == 1314, "GSM8K prompt count differs")
    _require(
        [row.get("question_index") for row in prompt_records if isinstance(row, Mapping)]
        == list(range(1314)),
        "GSM8K question indices are not exactly 0..1313",
    )
    _require(gsm8k.get("dataset_sha256") == INPUT_FILES["gsm8k-test.jsonl"], "GSM8K dataset hash binding differs")

    longbench = _load_json(artifacts / "longbench-manifest.json")
    _require(isinstance(longbench, Mapping), "LongBench manifest must be an object")

    transfer_path = input_root / "TRANSFER_RECEIPT.json"
    transfer = _load_json(transfer_path)
    _require(isinstance(transfer, Mapping), "input transfer receipt must be an object")
    _require(
        transfer.get("schema") == "gdn-sealed-sglang-input-transfer-v1"
        and transfer.get("status") == "PASS",
        "input transfer receipt schema/status differs",
    )
    _require(transfer.get("artifacts_manifest_sha256") == INPUT_MANIFEST_SHA256, "input transfer manifest binding differs")
    _require(transfer.get("model_revision") == MODEL_REVISION, "input transfer model revision differs")
    return {
        "root": str(input_root.resolve()),
        "artifact_manifest": manifest_record,
        "artifact_files": [
            {
                "path": name,
                "bytes": (artifacts / name).stat().st_size,
                "sha256": digest,
                "classification": (
                    "winner_dependent_authority"
                    if name == "source-authority.json"
                    else "reusable_sealed_input"
                ),
            }
            for name, digest in sorted(entries.items())
        ],
        "gsm8k": {"shots": 5, "prompt_count": 1314, "total_rows": 1319},
        "longbench_manifest_schema": longbench.get("schema"),
        "transfer_receipt": {
            "path": str(transfer_path.resolve()),
            "sha256": _sha256(transfer_path),
            "schema": transfer.get("schema"),
            "status": transfer.get("status"),
        },
    }


def _verify_model(model_root: Path) -> dict[str, object]:
    _require(model_root.is_absolute() and model_root.is_dir(), "model root must be an absolute directory")
    model_dir = model_root / "model"
    manifest_path = model_root / "evidence/binding/run/model-files.json"
    stage_receipt_path = model_root / "evidence/binding/run/receipt.json"
    transfer_path = model_root / "MODEL_TRANSFER_RECEIPT.json"
    verification_path = model_root / "MODEL_VERIFICATION_RESULT.json"

    manifest_record = _verified_file(manifest_path, MODEL_MANIFEST_SHA256)
    stage_record = _verified_file(stage_receipt_path, MODEL_STAGE_RECEIPT_SHA256)
    transfer_record = _verified_file(transfer_path, MODEL_TRANSFER_RECEIPT_SHA256)
    verification_record = _verified_file(
        verification_path, MODEL_VERIFICATION_RESULT_SHA256
    )
    rows = _load_json(manifest_path)
    _require(isinstance(rows, list) and len(rows) == MODEL_FILE_COUNT, "model manifest must be an 18-row list")
    names: set[str] = set()
    total_bytes = 0
    weight_bytes = 0
    weight_count = 0
    file_records = []
    for index, row in enumerate(rows):
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"},
            f"model manifest row {index} fields differ",
        )
        name, size, digest = row["path"], row["bytes"], row["sha256"]
        _require(
            isinstance(name, str)
            and name not in ("", ".", "..")
            and "/" not in name
            and "\\" not in name
            and name not in names,
            f"model manifest path is unsafe or duplicated at row {index}",
        )
        _require(type(size) is int and size >= 0, f"model byte count differs at row {index}")
        _require(isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None, f"model SHA256 differs at row {index}")
        path = model_dir / name
        _require(path.is_file() and not path.is_symlink(), f"sealed model file is missing: {name}")
        _require(path.stat().st_size == size, f"sealed model byte count differs: {name}")
        _require(_sha256(path) == digest, f"sealed model SHA256 differs: {name}")
        names.add(name)
        total_bytes += size
        if name.startswith("model-") and name.endswith(".safetensors"):
            weight_count += 1
            weight_bytes += size
        file_records.append({"path": name, "bytes": size, "sha256": digest})
    actual_names = {path.name for path in model_dir.iterdir()}
    _require(actual_names == names, "sealed model directory file set differs")
    _require(total_bytes == MODEL_BYTES, "sealed model total bytes differ")
    _require(weight_count == MODEL_WEIGHT_FILE_COUNT, "sealed model weight count differs")
    _require(weight_bytes == MODEL_WEIGHT_BYTES, "sealed model weight bytes differ")

    config = _load_json(model_dir / "config.json")
    _require(isinstance(config, Mapping), "model config must be an object")
    vocab_size = config.get("vocab_size")
    if vocab_size is None and isinstance(config.get("text_config"), Mapping):
        vocab_size = config["text_config"].get("vocab_size")
    _require(type(vocab_size) is int and vocab_size > 1, "model config vocabulary size is invalid")

    stage = _load_json(stage_receipt_path)
    transfer = _load_json(transfer_path)
    verification = _load_json(verification_path)
    _require(isinstance(stage, Mapping) and stage.get("status") == "PASS", "model stage receipt status differs")
    _require(stage.get("model_revision_observed") == MODEL_REVISION, "model stage revision differs")
    _require(stage.get("model_file_count") == MODEL_FILE_COUNT, "model stage file count differs")
    _require(
        isinstance(transfer, Mapping)
        and transfer.get("schema") == "gdn-final-sealed-model-transfer-receipt-v1"
        and transfer.get("status") == "PASS",
        "model transfer receipt schema/status differs",
    )
    transfer_model = transfer.get("model")
    _require(
        isinstance(transfer_model, Mapping)
        and transfer_model.get("repository") == MODEL_ID
        and transfer_model.get("revision") == MODEL_REVISION
        and transfer_model.get("manifest_sha256") == MODEL_MANIFEST_SHA256
        and transfer_model.get("file_count") == MODEL_FILE_COUNT
        and transfer_model.get("bytes") == MODEL_BYTES
        and transfer_model.get("weight_file_count") == MODEL_WEIGHT_FILE_COUNT
        and transfer_model.get("weight_bytes") == MODEL_WEIGHT_BYTES,
        "model transfer receipt binding differs",
    )
    _require(
        isinstance(verification, Mapping)
        and verification.get("status") == "PASS"
        and verification.get("model_file_count") == MODEL_FILE_COUNT
        and verification.get("model_manifest_sha256") == MODEL_MANIFEST_SHA256,
        "model verification receipt differs",
    )
    return {
        "root": str(model_root.resolve()),
        "model_path": str(model_dir.resolve()),
        "tokenizer_path": str(model_dir.resolve()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vocab_size": vocab_size,
        "file_count": MODEL_FILE_COUNT,
        "bytes": total_bytes,
        "weight_file_count": weight_count,
        "weight_bytes": weight_bytes,
        "manifest": manifest_record,
        "stage_receipt": stage_record,
        "transfer_receipt": transfer_record,
        "verification_result": verification_record,
        "files": file_records,
    }


def _verify_wheelhouse(wheelhouse: Path) -> dict[str, object]:
    _require(wheelhouse.is_absolute() and wheelhouse.is_dir(), "wheelhouse must be an absolute directory")
    manifest_path = wheelhouse / "wheels.manifest.json"
    receipt_path = wheelhouse / "receipt.json"
    seal_path = wheelhouse / "SEAL.sha256"
    wheel_sums_path = wheelhouse / "WHEELS.sha256"
    manifest_record = _verified_file(manifest_path, WHEEL_MANIFEST_SHA256)
    manifest = _load_json(manifest_path)
    receipt = _load_json(receipt_path)
    _require(
        isinstance(manifest, Mapping)
        and manifest.get("schema") == "gdn-gb300-runtime-dependency-wheel-manifest-v1"
        and manifest.get("target")
        == {
            "architecture": "aarch64",
            "cuda_major": 13,
            "manylinux_floor": "2_28",
            "python_abi": "cp312",
        }
        and isinstance(manifest.get("packages"), list)
        and len(manifest["packages"]) == 6,
        "wheel manifest schema/target/package count differs",
    )
    _require(
        isinstance(receipt, Mapping)
        and receipt.get("schema") == "gdn-gb300-runtime-dependency-wheelhouse-receipt-v1"
        and receipt.get("status") == "PASS_SEALED"
        and receipt.get("manifest_sha256") == WHEEL_MANIFEST_SHA256,
        "wheelhouse receipt differs",
    )
    seal_entries = _checksum_manifest(seal_path, wheelhouse)
    _require(
        {"wheels.manifest.json", "preflight.json", "receipt.json", "WHEELS.sha256", "stage_wheelhouse.sh"}
        <= set(seal_entries),
        "wheelhouse seal is incomplete",
    )
    wheel_entries = _checksum_manifest(wheel_sums_path, wheelhouse)
    expected_wheel_paths = {f"wheels/{name}" for name in WHEEL_FILENAMES}
    _require(set(wheel_entries) == expected_wheel_paths, "sealed wheel filename set differs")
    return {
        "root": str(wheelhouse.resolve()),
        "manifest": manifest_record,
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256(receipt_path),
            "schema": receipt.get("schema"),
            "status": receipt.get("status"),
        },
        "seal": {
            "path": str(seal_path.resolve()),
            "sha256": _sha256(seal_path),
            "entries": seal_entries,
        },
        "wheel_checksums": {
            "path": str(wheel_sums_path.resolve()),
            "sha256": _sha256(wheel_sums_path),
            "entries": wheel_entries,
        },
    }


def _environment() -> dict[str, object]:
    _require(platform.machine() == "aarch64", "preflight requires an aarch64 compute node")
    _require(bool(os.environ.get("SLURM_JOB_ID")), "preflight must run inside Slurm")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise PreflightError("pinned runtime torch is required for GPU authentication") from exc
    _require(torch.cuda.is_available(), "CUDA is not available")
    _require(torch.cuda.device_count() == 1, "preflight step must expose exactly one GPU")
    capability = list(torch.cuda.get_device_capability(0))
    gpu_name = torch.cuda.get_device_name(0)
    _require(capability == [10, 3], f"GPU compute capability {capability} is not [10,3]")
    _require("GB300" in gpu_name, f"GPU is not GB300: {gpu_name}")
    smi_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip().splitlines()
    _require(len(smi_query) == 1, "nvidia-smi must report exactly one visible GPU")
    smi_text = subprocess.run(
        ["nvidia-smi"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_text)
    _require(cuda_match is not None, "nvidia-smi did not report a CUDA version")
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": 1,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "nvidia_smi_gpu_row": smi_query[0],
        "driver_cuda_version": cuda_match.group(1),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _git_identity(root: Path) -> dict[str, str]:
    def git(spec: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", spec],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    _require(not status, "preflight source checkout must be completely clean")
    return {"commit": git("HEAD^{commit}"), "tree": git("HEAD^{tree}")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a fresh absolute file below an existing directory")
    digest_output = args.output.with_name(args.output.name + ".sha256")
    if digest_output.exists():
        parser.error("receipt SHA256 output must also be fresh")

    physical_start_ns = time.time_ns()
    measured_start_ns = time.time_ns()
    try:
        environment = _environment()
        inputs = _verify_inputs(args.input_root)
        model = _verify_model(args.model_root)
        wheelhouse = _verify_wheelhouse(args.wheelhouse)
        source = _git_identity(Path(__file__).resolve().parents[2])
    except (OSError, subprocess.CalledProcessError, PreflightError, ValueError) as exc:
        parser.error(str(exc))
    measured_finish_ns = time.time_ns()
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_REUSABLE_ASSETS_ONLY",
        "claim_scope": "reusable_inputs_model_and_runtime_dependencies_only",
        "final_winner_or_campaign_evidence": False,
        "source": source,
        "environment": environment,
        "inputs": inputs,
        "model": model,
        "wheelhouse": wheelhouse,
        "timing": {
            "physical_started_at": datetime.fromtimestamp(
                physical_start_ns / 1_000_000_000, tz=timezone.utc
            ).isoformat(),
            "finished_at": datetime.fromtimestamp(
                measured_finish_ns / 1_000_000_000, tz=timezone.utc
            ).isoformat(),
            "physical_turnaround_seconds": (measured_finish_ns - physical_start_ns)
            / 1_000_000_000,
            "measured_runtime_seconds": (measured_finish_ns - measured_start_ns)
            / 1_000_000_000,
        },
    }
    encoded = (json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(encoded)
    receipt_sha256 = hashlib.sha256(encoded).hexdigest()
    with digest_output.open("x", encoding="ascii") as handle:
        handle.write(f"{receipt_sha256}  {args.output.name}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "receipt_sha256": receipt_sha256,
                "schema": SCHEMA,
                "status": "PASS_REUSABLE_ASSETS_ONLY",
                "vocab_size": model["vocab_size"],
                "measured_runtime_seconds": receipt["timing"]["measured_runtime_seconds"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
