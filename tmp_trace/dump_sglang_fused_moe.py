"""
Benchmark script for ShareGPT dataset using different models with SGLang.

Usage:
    # Using Llama model (auto TP size = 1)
    python3 bench_sharegpt.py --model llama --batch-size 16

    # Using DeepSeek model (TP size = 8, EP size = 8)
    python3 bench_sharegpt.py --model deepseek --batch-size 16

    # Using Qwen model (auto TP size = 1)
    python3 bench_sharegpt.py --model qwen --batch-size 16

    # If you are collecting ragged prefill (disables radix cache)
    python3 bench_sharegpt.py --model llama --batch-size 32 --ragged

Arguments:
--model: Model type to use (llama, deepseek, or qwen). This determines the TP configuration and model path.
--batch-size: Batch size (1, 2, 4, 8, 16, 32, 64, or 128). (1,16,64)
--ragged: Enable ragged batching (adds --disable-radix-cache to server args).
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

from datasets import load_dataset

from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


@dataclass
class TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    image_data: Optional[str] = None


def get_model_config(model_type: str) -> dict:
    """Get model-specific configuration including TP size, model path, and other parameters."""
    configs = {
        "llama": {
            "model_path": "/raid/catalyst/models/Llama-3.1-8B-Instruct",
            "tp_size": None,  # Auto TP size
            "ep_size": None,
            "quantization": "fp8",
            "use_flashinfer": True,
            "moe_backend": None,
        },
        "deepseek": {
            "model_path": "/raid/catalyst/models/DeepSeek-V3",
            "tp_size": 8,  # Fixed TP size for DeepSeek
            "ep_size": 8,  # Fixed EP size for DeepSeek
            "quantization": "fp8",
            "use_flashinfer": True,
            "moe_backend": "flashinfer_trtllm",
        },
        "qwen": {
            "model_path": "/raid/catalyst/models/Qwen3-30B-A3B",
            "tp_size": None,  # Auto TP size
            "ep_size": None,
            "quantization": "fp8",
            "use_flashinfer": True,
            "moe_backend": None,
        },
    }

    if model_type.lower() not in configs:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: {list(configs.keys())}"
        )

    return configs[model_type.lower()]


def log(msg):
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def load_prompts_from_sharegpt(n: int) -> List[str]:
    """Load n prompts from the ShareGPT dataset."""
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True,
    )
    prompts = []

    for example in ds:
        conv = example.get("conversations", [])
        if conv and conv[0]["from"].lower() == "human":
            prompts.append(conv[0]["value"])
        if len(prompts) >= n:
            break

    log(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
    return prompts


class DummyTokenizer:
    """Dummy tokenizer for compatibility with benchmark function."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def run_benchmark(
    base_url: str,
    prompts: list,
    batch_size: int,
    label: str = "",
) -> list:
    """Run benchmark with given prompts and return results."""
    tokenizer = DummyTokenizer()
    set_global_args(
        SimpleNamespace(
            disable_ignore_eos=False,
            disable_stream=True,
            return_logprob=False,
            backend="sglang",
            dataset_name="custom",
            num_prompts=None,
            sharegpt_output_len=None,
            random_input_len=None,
            random_output_len=None,
            random_range_ratio=None,
            output_file=None,
            output_details=False,
            warmup_requests=1,
        )
    )

    num_batches = math.ceil(len(prompts) / batch_size)
    all_results = []

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        input_requests = [
            TestRequest(prompt=p, prompt_len=0, output_len=512) for p in batch_prompts
        ]

        extra_request_body = {}
        extra_request_body["timestamp"] = time.time()



        log(f"Running batch {i + 1}/{num_batches} with {len(input_requests)} prompts")
        results = asyncio.run(
            benchmark(
                backend="sglang",
                api_url=f"{base_url}/generate",
                base_url=base_url,
                model_id="default",
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=float("inf"),
                max_concurrency=batch_size,
                disable_tqdm=False,
                lora_names=None,
                extra_request_body={
                    "sampling_params": {"temperature": 0.7, **extra_request_body}
                },
                profile=None,
            )
        )
        all_results.append(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ShareGPT dataset with SGLang using different models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bench_sharegpt.py --model llama
  python3 bench_sharegpt.py --model deepseek --batch-size 128
  python3 bench_sharegpt.py --model qwen --ragged
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama", "deepseek", "qwen"],
        help="Model type (determines TP configuration and model path): llama, deepseek, or qwen",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        choices=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch size - must be one of: 1, 2, 4, 8, 16, 32, 64, 128 (default: 64).",
    )
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Enable ragged batching (adds --disable-radix-cache to server arguments)",
    )
    args = parser.parse_args()

    # Get model configuration
    model_config = get_model_config(args.model)
    # batch_sizes = [1, 16, 64]
    batch_sizes = [args.batch_size]
    num_batches = 1
    num_prompts = 64
    base_url = "http://127.0.0.1:20000"

    log(f"Starting benchmark for {args.model} model")
    log(f"Total prompts: {num_prompts} ({num_batches} batches)")
    log(f"Ragged batching: {'enabled' if args.ragged else 'disabled'}")
    log(f"Model path: {model_config['model_path']}")
    log(f"Model config: {model_config}")

    # Load ShareGPT prompts
    prompts = load_prompts_from_sharegpt(num_prompts)

    # Build server arguments
    server_args = [
        "--disable-cuda-graph",
        "--quantization",
        model_config["quantization"],
        "--trust-remote-code",
    ]

    # Add disable-radix-cache if ragged is enabled
    if args.ragged:
        server_args.append("--disable-radix-cache")
        log("Added --disable-radix-cache (ragged batching enabled)")

    # Add TP size if specified (otherwise use auto)
    if model_config["tp_size"] is not None:
        server_args.extend(["--tp-size", str(model_config["tp_size"])])
        log(f"Using TP size: {model_config['tp_size']}")
    else:
        log("Using auto TP size")

    # Add attention backend if supported
    # if model_config["use_flashinfer"]:
    #     server_args.extend(["--attention-backend", "flashinfer"])

    # Add fused_moe arguments for deepseek
    if model_config["moe_backend"] is not None:
        server_args.extend(["--moe-runner-backend", model_config["moe_backend"]])

    if model_config["ep_size"] is not None:
        server_args.extend(["--ep-size", str(model_config["ep_size"])])

    log(f"Server arguments: {server_args}")

    # Launch the server
    process = popen_launch_server(
        model_config["model_path"],
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            **os.environ,
        },
    )

    try:
        for batch_size in batch_sizes:

            log(
                f"Running benchmark with batch size {batch_size}"
            )
            results = run_benchmark(
                base_url, prompts[:batch_size], batch_size
            )
            log(
                f"Completed benchmark for batch size {batch_size}"
            )

    except Exception as e:
        log(f"Benchmark failed: {e}")
        raise
    finally:
        log("Shutting down server...")
        kill_process_tree(process.pid)
        time.sleep(3)
        log("Server shutdown complete")


if __name__ == "__main__":
    main()
