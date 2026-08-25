#!/usr/bin/env python3
"""Fail closed on source, wheel, API, container, cluster, and GPU identity."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import socket
import subprocess
from pathlib import Path

import torch
from packaging.version import Version


EXPECTED_CANDIDATE_PYTHON_VERSION = "0.6.18"
EXPECTED_IMAGE_CUBIN_VERSION = "0.6.14"
EXPECTED_CAKE_SMEM_BYTES = 197632
EXPECTED_CAKE_KERNEL_SYMBOL = (
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4"
)


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "@sha256:" not in args.container_image:
        raise RuntimeError("Container image is not pinned by sha256 digest")
    version_check_bypass = os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK")
    if version_check_bypass != "1":
        raise RuntimeError(
            "FLASHINFER_DISABLE_VERSION_CHECK must be exactly 1 for the pinned "
            "image cubin/candidate Python version split"
        )
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
    cubin_distribution = importlib.metadata.distribution("flashinfer-cubin")
    distribution_root = Path(distribution.locate_file("")).resolve()
    cubin_distribution_root = Path(cubin_distribution.locate_file("")).resolve()
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
    if distribution.version != EXPECTED_CANDIDATE_PYTHON_VERSION:
        raise RuntimeError(
            f"Candidate flashinfer-python {distribution.version} != "
            f"{EXPECTED_CANDIDATE_PYTHON_VERSION}"
        )
    cubin_base_version = Version(cubin_distribution.version).base_version
    if cubin_base_version != EXPECTED_IMAGE_CUBIN_VERSION:
        raise RuntimeError(
            f"Image flashinfer-cubin base version {cubin_base_version} != "
            f"{EXPECTED_IMAGE_CUBIN_VERSION}"
        )

    expected_api_source = (
        distribution_root
        / "flashinfer/comm/all_gather_matmul/all_gather_matmul.py"
    )
    cake_backend_source = (
        distribution_root
        / "flashinfer/comm/all_gather_matmul/cake_all_gather_matmul.py"
    )
    cake_kernel_source = (
        distribution_root
        / "flashinfer/data/csrc/cake_all_gather_matmul/sm100a/"
        "cake_all_gather_matmul_kernels.cu"
    )
    if api_source_path != expected_api_source:
        raise RuntimeError(
            f"all_gather_matmul API source {api_source_path} != {expected_api_source}"
        )
    for source in (cake_backend_source, cake_kernel_source):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Candidate Cake source is missing or a symlink: {source}")
        if distribution_root not in source.resolve().parents:
            raise RuntimeError(f"Candidate Cake source escaped fresh target: {source}")
    backend_text = cake_backend_source.read_text()
    kernel_text = cake_kernel_source.read_text()
    if EXPECTED_CAKE_KERNEL_SYMBOL not in backend_text:
        raise RuntimeError("Candidate Cake backend does not select the exact ws4 symbol")
    if EXPECTED_CAKE_KERNEL_SYMBOL not in kernel_text:
        raise RuntimeError("Candidate Cake kernel source lacks the exact ws4 symbol")
    expected_smem_define = f"#define SMEM_TOTAL {EXPECTED_CAKE_SMEM_BYTES}"
    if expected_smem_define not in kernel_text:
        raise RuntimeError(
            f"Candidate Cake kernel source lacks {expected_smem_define!r}"
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
    allowed_gpu_names = {"NVIDIA GB200", "NVIDIA B200"}
    for index, gpu in enumerate(gpus):
        if gpu["name"] not in allowed_gpu_names:
            raise RuntimeError(f"GPU name is not in the GB200/B200 allow-set: {gpu}")
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
            "python_distribution_version": distribution.version,
            "cubin_distribution_version": cubin_distribution.version,
            "cubin_distribution_base_version": cubin_base_version,
            "cubin_distribution_root": str(cubin_distribution_root),
            "version_check_bypass": {
                "environment_variable": "FLASHINFER_DISABLE_VERSION_CHECK",
                "value": version_check_bypass,
                "scope": (
                    "pinned image flashinfer-cubin 0.6.14 with candidate "
                    "flashinfer-python 0.6.18"
                ),
            },
            "cake_backend_source_path": str(cake_backend_source),
            "cake_backend_source_sha256": sha256_file(cake_backend_source),
            "cake_kernel_source_path": str(cake_kernel_source),
            "cake_kernel_source_sha256": sha256_file(cake_kernel_source),
            "cake_kernel_symbol": EXPECTED_CAKE_KERNEL_SYMBOL,
            "cake_dynamic_smem_bytes": EXPECTED_CAKE_SMEM_BYTES,
        },
        "sglang_import_path": str(sglang_import_path),
        "gpus": gpus,
        "allowed_gpu_names": sorted(allowed_gpu_names),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
