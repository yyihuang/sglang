#!/usr/bin/env python3
"""Atomically create arm and run completion receipts with immutable bindings."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"Short write while creating completion receipt {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("arm", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--variant", choices=("native", "explicit", "candidate"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--input-contract", type=Path)
    args = parser.parse_args()

    runtime = json.loads(args.runtime_contract.read_text())
    payload = {
        "status": "pass",
        "kind": args.kind,
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_contract_sha256": sha256_file(args.runtime_contract),
        "slurm_job_id": runtime["slurm_job_id"],
        "container_image": runtime["container_image"],
        "sglang_commit": runtime["sglang"]["commit"],
        "sglang_tree": runtime["sglang"]["tree"],
        "flashinfer_commit": runtime["flashinfer"]["commit"],
        "flashinfer_tree": runtime["flashinfer"]["tree"],
        "flashinfer_wheel_sha256": runtime["flashinfer"]["wheel_sha256"],
        "flashinfer_api_signature": runtime["flashinfer"]["api_signature"],
    }
    if args.kind == "arm":
        if args.variant is None or args.summary is not None or args.input_contract is not None:
            raise RuntimeError("Arm receipt requires only --variant")
        payload["variant"] = args.variant
    else:
        if args.variant is not None or args.summary is None or args.input_contract is None:
            raise RuntimeError("Run receipt requires --summary and --input-contract")
        summary = json.loads(args.summary.read_text())
        if summary.get("status") != "pass":
            raise RuntimeError("Cannot complete a run whose summary did not pass")
        payload.update(
            {
                "summary_sha256": sha256_file(args.summary),
                "artifact_inventory_sha256": summary["artifact_inventory_sha256"],
                "input_contract_sha256": sha256_file(args.input_contract),
            }
        )
    write_exclusive(args.output, payload)


if __name__ == "__main__":
    main()
