"""
Benchmark script for ShareGPT dataset using Qwen3-Next with SGLang.
Usage:
  python3 run_qwen_next.py
"""

import argparse
import asyncio
import math
import os
import time
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import kill_process_tree, popen_launch_server


# -----------------------------
# Workload / request structures
# -----------------------------

@dataclass
class TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    extra_request_body: Dict[str, Any] = field(default_factory=dict)
    routing_key: Optional[str] = None


class DummyTokenizer:
    """Dummy tokenizer for compatibility with benchmark()."""
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def log(msg: str) -> None:
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


# -----------------------------
# ShareGPT loader
# -----------------------------

def load_prompts_from_sharegpt(n: int) -> List[str]:
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True,
    )

    prompts: List[str] = []
    for example in ds:
        conv = example.get("conversations", [])
        if conv and conv[0].get("from", "").lower() == "human":
            prompts.append(conv[0].get("value", ""))
        if len(prompts) >= n:
            break

    log(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
    return prompts


# -----------------------------
# bench_serving args compat
# -----------------------------

def ensure_bench_serving_args(ns: SimpleNamespace) -> SimpleNamespace:
    """
    sglang.bench_serving expects a module-global `args` with these attributes.
    Ensure they exist with safe defaults.
    """
    defaults = {
        # streaming / eos
        "disable_stream": False,
        "disable_ignore_eos": False,

        # return extras
        "return_logprob": False,
        "return_routed_experts": False,

        # headers
        "header": None,

        # warmup / dataset / plotting
        "warmup_requests": 3,
        "dataset_name": "custom",
        "plot_throughput": False,

        # profiler knobs (safe defaults even if profile=False)
        "profile_activities": ["CPU", "GPU"],
        "profile_num_steps": None,
        "profile_by_stage": False,
        "profile_stages": None,
    }

    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


# -----------------------------
# Benchmark runner
# -----------------------------

def run_benchmark(base_url: str, prompts: List[str], batch_size: int) -> List[Any]:
    tokenizer = DummyTokenizer()

    bench_args = SimpleNamespace(
        backend="sglang",

        # required / requested args
        disable_stream=True,
        disable_ignore_eos=False,
        return_logprob=False,
        return_routed_experts=False,
        header=None,
        warmup_requests=3,
        dataset_name="custom",
        plot_throughput=False,

        # optional extras used by bench_serving in some paths
        output_file=None,
        output_details=False,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
    )
    set_global_args(ensure_bench_serving_args(bench_args))

    num_batches = math.ceil(len(prompts) / batch_size)
    all_results: List[Any] = []

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        t0 = time.time()

        input_requests = [
            TestRequest(
                prompt=p,
                prompt_len=0,
                output_len=0,
                timestamp=t0,
                text_prompt_len=0,
                vision_prompt_len=0,
                image_data=None,
                extra_request_body={},
                routing_key=None,
            )
            for p in batch_prompts
        ]

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
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={"sampling_params": {"temperature": 0}},
                profile=False,
            )
        )
        all_results.append(results)

    return all_results


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    env = os.environ.copy()
    trace_root = os.getcwd()
    env["SGLANG_FLASHINFER_GDN_WORKLOAD_DIR"] = os.path.join(trace_root, "gdn")
    env.setdefault("SGLANG_FLASHINFER_GDN_WORKLOAD_LIMIT", "10")
    parser = argparse.ArgumentParser(
        description="Benchmark ShareGPT with Qwen3-Next on SGLang",
    )
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=3)
    args = parser.parse_args()

    model_path = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    base_url = f"http://{args.host}:{args.port}"

    # Prompts
    num_prompts = args.batch_size * args.num_batches
    prompts = load_prompts_from_sharegpt(num_prompts)

    # Server flags: Qwen3-Next (match launch_server command: python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 8 --attention-backend flashinfer)
    server_args = [
        "--tp", "8",
        "--attention-backend", "flashinfer",
    ]

    log(f"Launching server: model_path={model_path}")
    log(f"Server arguments: {server_args}")

    process = popen_launch_server(
        model_path,
        base_url,
        timeout=1800,
        other_args=server_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            "SGLANG_TEST_REQUEST_TIME_STATS": "1",
            **env,
        },
    )

    try:
        log(f"Running benchmark with batch size {args.batch_size}")
        # Use user-specified warmup_requests for bench_serving.args
        _ = run_benchmark(base_url, prompts[: args.batch_size * args.num_batches], args.batch_size)

    finally:
        log("Shutting down server...")
        kill_process_tree(process.pid)
        time.sleep(3)
        log("Server shutdown complete")


if __name__ == "__main__":
    main()