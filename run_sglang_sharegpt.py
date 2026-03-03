"""
SGLang ShareGPT benchmark script.

Launches an SGLang server with the given model and server args,
then runs the ShareGPT benchmark.

Usage:
  python run_sglang_sharegpt.py --model <path> [opts] [-- server_arg ...]

Everything after '--' is forwarded verbatim to the SGLang server.

Examples:
  python run_sglang_sharegpt.py --model deepseek-ai/DeepSeek-V3.2

  python run_sglang_sharegpt.py \\
      --model /data/models/DeepSeek-V3.2 \\
      --batch-size 64 --num-batches 2 \\
      -- --tp 4 --dp 4 --enable-dp-attention \\
         --mem-fraction-static 0.7 --page-size 64
"""

import argparse
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import kill_process_tree, popen_launch_server


# ── Request structures ────────────────────────────────────────────────────────

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
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def log(msg: str) -> None:
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


# ── ShareGPT loader ───────────────────────────────────────────────────────────

def load_sharegpt_prompts(n: int, dataset_path: str = "") -> List[str]:
    if dataset_path:
        ds = load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            streaming=True,
        )
    else:
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


# ── bench_serving compat ──────────────────────────────────────────────────────

def make_bench_args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_stream=True,
        disable_ignore_eos=False,
        return_logprob=False,
        return_routed_experts=False,
        header=None,
        warmup_requests=3,
        dataset_name="custom",
        plot_throughput=False,
        profile_activities=["CPU", "GPU"],
        profile_num_steps=None,
        profile_by_stage=False,
        profile_stages=None,
        output_file=None,
        output_details=False,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
    )


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(base_url: str, prompts: List[str], batch_size: int) -> List[Any]:
    set_global_args(make_bench_args())

    tokenizer = DummyTokenizer()
    num_batches = math.ceil(len(prompts) / batch_size)
    all_results: List[Any] = []

    for i in range(num_batches):
        batch = prompts[i * batch_size : (i + 1) * batch_size]
        t0 = time.time()

        input_requests = [
            TestRequest(
                prompt=p,
                prompt_len=0,
                output_len=0,
                timestamp=t0,
                text_prompt_len=0,
                vision_prompt_len=0,
            )
            for p in batch
        ]

        log(f"Batch {i + 1}/{num_batches} — {len(input_requests)} prompts")

        result = asyncio.run(
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
        all_results.append(result)

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _split_on_dashdash(argv):
    try:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    except ValueError:
        return argv, []


def main() -> None:
    script_argv, server_args = _split_on_dashdash(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Launch SGLang server and run ShareGPT benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face hub ID.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=1800, help="Server startup timeout (s).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--dataset-path", default="", help="Local path to ShareGPT JSON file.")
    parser.add_argument(
        "--skip-launch", action="store_true",
        help="Skip server launch and connect to an already-running server.",
    )
    args = parser.parse_args(script_argv)

    base_url = f"http://{args.host}:{args.port}"
    num_prompts = args.batch_size * args.num_batches
    prompts = load_sharegpt_prompts(num_prompts, args.dataset_path)

    process = None
    if not args.skip_launch:
        log(f"Launching server: model={args.model}")
        log(f"Server args: {server_args}")
        process = popen_launch_server(
            args.model,
            base_url,
            timeout=args.timeout,
            other_args=server_args,
        )

    try:
        log(f"Running benchmark: batch_size={args.batch_size}, num_batches={args.num_batches}")
        run_benchmark(base_url, prompts, args.batch_size)
    finally:
        if process is not None:
            log("Shutting down server...")
            kill_process_tree(process.pid)
            time.sleep(3)
            log("Server shutdown complete.")


if __name__ == "__main__":
    main()
