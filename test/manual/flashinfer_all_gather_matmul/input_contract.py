#!/usr/bin/env python3
"""Verify pinned inputs and hash the exact deterministic request selection."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from sglang.benchmark.datasets.random import sample_random_requests


GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
MODEL_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"


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
    manifest_payload = json.dumps(weights, separators=(",", ":"), sort_keys=True).encode()
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_sha != model_result["weight_manifest_sha256"]:
        raise RuntimeError("Weight manifest digest does not match model result")

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
