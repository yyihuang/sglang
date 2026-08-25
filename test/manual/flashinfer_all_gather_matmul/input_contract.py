#!/usr/bin/env python3
"""Verify pinned inputs and hash the exact deterministic request selection."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from sglang.benchmark.datasets.random import sample_random_requests


GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
MODEL_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"
MODEL_MANIFEST_SHA256 = "438a360fe1b9c1f748fdd543757f11eb677c453cc522aa61100c4a5e6dce2c6f"
MODEL_CONFIG_SHA256 = "fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749"
MODEL_INDEX_SHA256 = "2abe0910e23770a30ccf9b1b91804c64831c47f9c98defaa5293aa999433fc2b"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--sharegpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gsm8k_sha = sha256_file(args.gsm8k)
    sharegpt_sha = sha256_file(args.sharegpt)
    if gsm8k_sha != GSM8K_SHA256 or sharegpt_sha != SHAREGPT_SHA256:
        raise RuntimeError(
            f"Dataset digest mismatch: gsm8k={gsm8k_sha}, sharegpt={sharegpt_sha}"
        )
    model_result = json.loads(args.model_result.read_text())
    weights = json.loads(args.weights_manifest.read_text())
    if model_result["revision"] != MODEL_REVISION or model_result["resolved_sha"] != MODEL_REVISION:
        raise RuntimeError(f"Unexpected model revision: {model_result}")
    if len(weights) != 30:
        raise RuntimeError(f"Expected 30 weight shards, found {len(weights)}")
    if Path(model_result["model_path"]).resolve() != args.model_path.resolve():
        raise RuntimeError(
            f"MODEL_PATH {args.model_path.resolve()} != receipt {model_result['model_path']}"
        )
    manifest_payload = json.dumps(weights, separators=(",", ":"), sort_keys=True).encode()
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError(f"Unexpected pinned model manifest: {manifest_sha}")
    if manifest_sha != model_result["weight_manifest_sha256"]:
        raise RuntimeError("Weight manifest digest does not match model result")

    verify_started = time.time()

    def verify_weight(row):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe weight path in manifest: {relative}")
        path = args.model_path / relative
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != row["size"] or digest != row["sha256"]:
            raise RuntimeError(f"Runtime weight mismatch: {relative}")
        return {"path": str(relative), "size": size, "sha256": digest}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified_weights = list(pool.map(verify_weight, weights))

    config_path = args.model_path / "config.json"
    index_path = args.model_path / "model.safetensors.index.json"
    config_sha = sha256_file(config_path)
    index_sha = sha256_file(index_path)
    if config_sha != MODEL_CONFIG_SHA256 or index_sha != MODEL_INDEX_SHA256:
        raise RuntimeError(
            f"Runtime model metadata mismatch: config={config_sha}, index={index_sha}"
        )
    config = json.loads(config_path.read_text())
    shape_contract = {
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "num_hidden_layers": config["num_hidden_layers"],
        "torch_dtype": config.get("torch_dtype"),
    }
    if shape_contract != model_result["shape_contract"]:
        raise RuntimeError(f"Runtime model config differs from receipt: {shape_contract}")
    index = json.loads(index_path.read_text())
    indexed_shards = sorted(set(index["weight_map"].values()))
    if indexed_shards != sorted(row["path"] for row in weights):
        raise RuntimeError("Safetensors index does not reference the pinned 30 shards")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    random.seed(20260825)
    np.random.seed(20260825)
    requests = sample_random_requests(
        input_len=4096,
        output_len=128,
        num_prompts=256,
        range_ratio=0.0,
        tokenizer=tokenizer,
        dataset_path=str(args.sharegpt),
        random_sample=True,
        return_text=True,
    )
    selected = [
        {
            "prompt_sha256": hashlib.sha256(row.prompt.encode()).hexdigest(),
            "prompt_len": row.prompt_len,
            "output_len": row.output_len,
        }
        for row in requests
    ]
    selection_payload = json.dumps(selected, separators=(",", ":"), sort_keys=True).encode()
    result = {
        "model_repo": "meta-llama/Llama-3.1-70B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_lfs_manifest_sha256": manifest_sha,
        "model_weight_bytes": model_result["weight_bytes"],
        "model_weight_shards": len(weights),
        "model_config_sha256": config_sha,
        "model_safetensors_index_sha256": index_sha,
        "runtime_weight_verification_seconds": time.time() - verify_started,
        "verified_weights": verified_weights,
        "gsm8k": {
            "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
            "sha256": gsm8k_sha,
            "selection": "first 5 shots, next 500 evaluation rows; next 32 for parity",
        },
        "sharegpt": {
            "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "revision": "192ab2185289094fc556ec8ce5ce1e8e587154ca",
            "file": "ShareGPT_V3_unfiltered_cleaned_split.json",
            "sha256": sharegpt_sha,
            "serving_selection_sha256": hashlib.sha256(selection_payload).hexdigest(),
            "selected_requests": selected,
        },
        "seed": 20260825,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
