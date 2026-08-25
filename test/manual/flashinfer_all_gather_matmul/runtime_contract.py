#!/usr/bin/env python3
"""Fail closed on source, wheel, API, container, cluster, and GPU identity."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import socket
import subprocess
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--sglang-tree", required=True)
    parser.add_argument("--flashinfer-wheel", type=Path, required=True)
    parser.add_argument("--flashinfer-receipt", type=Path, required=True)
    parser.add_argument("--flashinfer-commit", required=True)
    parser.add_argument("--flashinfer-tree", required=True)
    parser.add_argument("--flashinfer-wheel-sha256", required=True)
    parser.add_argument("--flashinfer-api-signature", required=True)
    parser.add_argument("--flashinfer-install-root", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--expected-cluster", required=True)
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "@sha256:" not in args.container_image:
        raise RuntimeError("Container image is not pinned by sha256 digest")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if cluster != args.expected_cluster:
        raise RuntimeError(f"Cluster {cluster!r} != expected {args.expected_cluster!r}")

    actual_commit = git(args.sglang_root, "rev-parse", "HEAD")
    actual_tree = git(args.sglang_root, "rev-parse", "HEAD^{tree}")
    status = git(args.sglang_root, "status", "--porcelain=v1")
    if actual_commit != args.sglang_commit or actual_tree != args.sglang_tree or status:
        raise RuntimeError(
            f"SGLang source mismatch: commit={actual_commit}, tree={actual_tree}, status={status!r}"
        )

    receipt = json.loads(args.flashinfer_receipt.read_text())
    expected_receipt = {
        "commit": args.flashinfer_commit,
        "tree": args.flashinfer_tree,
        "wheel_sha256": args.flashinfer_wheel_sha256,
        "api_signature": args.flashinfer_api_signature,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"FlashInfer receipt {key}={receipt.get(key)!r}, expected {expected!r}")
    wheel_sha = sha256_file(args.flashinfer_wheel)
    if wheel_sha != args.flashinfer_wheel_sha256:
        raise RuntimeError(f"FlashInfer wheel SHA256 mismatch: {wheel_sha}")

    import flashinfer
    import sglang
    from flashinfer.comm import all_gather_matmul

    api_signature = str(inspect.signature(all_gather_matmul))
    if api_signature != args.flashinfer_api_signature:
        raise RuntimeError(
            f"FlashInfer API signature {api_signature!r} != {args.flashinfer_api_signature!r}"
        )
    if "backend" not in inspect.signature(all_gather_matmul).parameters:
        raise RuntimeError("FlashInfer all_gather_matmul has no backend parameter")
    distribution = importlib.metadata.distribution("flashinfer-python")
    distribution_root = Path(distribution.locate_file("")).resolve()
    import_path = Path(flashinfer.__file__).resolve()
    api_source_path = Path(inspect.getsourcefile(all_gather_matmul)).resolve()
    install_root = args.flashinfer_install_root.resolve()
    sglang_import_path = Path(sglang.__file__).resolve()
    sglang_python_root = (args.sglang_root / "python").resolve()
    if distribution_root != install_root:
        raise RuntimeError(
            f"FlashInfer distribution root {distribution_root} != isolated install {install_root}"
        )
    if distribution_root not in import_path.parents or distribution_root not in api_source_path.parents:
        raise RuntimeError(
            f"FlashInfer import escaped installed distribution: {import_path}, {api_source_path}"
        )
    if sglang_python_root not in sglang_import_path.parents:
        raise RuntimeError(
            f"SGLang import {sglang_import_path} escaped exact source {sglang_python_root}"
        )

    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line}")
        gpus.append(
            dict(
                zip(
                    ("index", "name", "uuid", "compute_cap", "pci_bus_id", "memory_mib"),
                    fields,
                )
            )
        )
    if len(gpus) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError(f"Expected four visible GPUs, nvidia-smi={len(gpus)}, torch={torch.cuda.device_count()}")
    if len({gpu["uuid"] for gpu in gpus}) != 4:
        raise RuntimeError(f"GPU UUIDs are not unique: {gpus}")
    name_pattern = re.compile(args.expected_gpu_name_regex)
    for index, gpu in enumerate(gpus):
        if not name_pattern.fullmatch(gpu["name"]):
            raise RuntimeError(f"GPU name does not match GB200 contract: {gpu}")
        if gpu["compute_cap"] != "10.0" or torch.cuda.get_device_capability(index) != (10, 0):
            raise RuntimeError(f"GPU is not SM100: {gpu}")

    result = {
        "hostname": socket.gethostname(),
        "slurm_cluster_name": cluster,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "container_image": args.container_image,
        "sglang": {
            "root": str(args.sglang_root.resolve()),
            "commit": actual_commit,
            "tree": actual_tree,
            "clean": True,
        },
        "flashinfer": {
            **expected_receipt,
            "wheel_path": str(args.flashinfer_wheel.resolve()),
            "distribution_version": distribution.version,
            "distribution_root": str(distribution_root),
            "import_path": str(import_path),
            "api_source_path": str(api_source_path),
        },
        "sglang_import_path": str(sglang_import_path),
        "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
