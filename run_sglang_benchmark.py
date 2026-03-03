"""
SGLang inference benchmark: launch server and run ShareGPT benchmark.

Usage:
  python run_sglang_benchmark.py --model <path> [benchmark opts] [-- server_arg ...]

Everything after '--' is forwarded verbatim to the SGLang server.

Examples:
  # Basic run with default settings
  python run_sglang_benchmark.py --model deepseek-ai/DeepSeek-V3.2

  # Custom benchmark settings + server args
  python run_sglang_benchmark.py \\
      --model /data/models/DeepSeek-V3.2 \\
      --num-prompts 500 \\
      --max-concurrency 64 \\
      -- --tp 4 --dp 4 --enable-dp-attention --mem-fraction-static 0.7

  # Connect to an already-running server (skip launch)
  python run_sglang_benchmark.py \\
      --model /data/models/DeepSeek-V3.2 \\
      --host 127.0.0.1 --port 30000 \\
      --skip-launch
"""

import sys
import time
import argparse
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.test.test_utils import kill_process_tree, popen_launch_server


def _split_on_dashdash(argv):
    """Split argv into script args and server args on the '--' separator."""
    try:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    except ValueError:
        return argv, []


def main():
    script_argv, server_args = _split_on_dashdash(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Launch an SGLang server and run a ShareGPT benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Server / launch ──────────────────────────────────────────────────────
    launch = parser.add_argument_group("server / launch")
    launch.add_argument(
        "--model", required=True, help="Model path or Hugging Face hub ID."
    )
    launch.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1).")
    launch.add_argument("--port", type=int, default=30000, help="Server port (default: 30000).")
    launch.add_argument(
        "--timeout", type=int, default=1800,
        help="Seconds to wait for the server to become ready (default: 1800).",
    )
    launch.add_argument(
        "--skip-launch", action="store_true",
        help="Skip server launch and connect to an already-running server.",
    )

    # ── Benchmark ────────────────────────────────────────────────────────────
    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--dataset-path", default="",
        help="Local path to the ShareGPT JSON file (leave empty to stream from HF hub).",
    )
    bench.add_argument(
        "--num-prompts", type=int, default=1000,
        help="Number of prompts to benchmark (default: 1000).",
    )
    bench.add_argument(
        "--sharegpt-output-len", type=int, default=None,
        help="Override output token length for every request.",
    )
    bench.add_argument(
        "--sharegpt-context-len", type=int, default=None,
        help="Drop requests whose total length exceeds this value.",
    )
    bench.add_argument(
        "--request-rate", type=float, default=float("inf"),
        help="Requests per second. 'inf' sends all requests at t=0 (default: inf).",
    )
    bench.add_argument(
        "--max-concurrency", type=int, default=None,
        help="Cap on in-flight concurrent requests.",
    )
    bench.add_argument(
        "--output-file", type=str, default=None,
        help="Write benchmark results to this JSONL file (default: auto-named).",
    )
    bench.add_argument(
        "--warmup-requests", type=int, default=3,
        help="Number of warmup requests before timing starts (default: 3).",
    )
    bench.add_argument(
        "--disable-stream", action="store_true",
        help="Disable streaming mode (measure end-to-end latency only).",
    )

    args = parser.parse_args(script_argv)

    base_url = f"http://{args.host}:{args.port}"

    # ── Launch server ─────────────────────────────────────────────────────────
    process = None
    if not args.skip_launch:
        print(f"[LAUNCHER] model     = {args.model}")
        print(f"[LAUNCHER] base_url  = {base_url}")
        print(f"[LAUNCHER] server_args = {server_args}")
        process = popen_launch_server(
            args.model,
            base_url,
            timeout=args.timeout,
            other_args=server_args,
        )

    # ── Run benchmark ─────────────────────────────────────────────────────────
    try:
        bench_args = SimpleNamespace(
            # connection
            backend="sglang",
            base_url=base_url,
            host=args.host,
            port=args.port,
            model=args.model,
            tokenizer=None,
            served_model_name=None,
            # dataset
            dataset_name="sharegpt",
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            sharegpt_output_len=args.sharegpt_output_len,
            sharegpt_context_len=args.sharegpt_context_len,
            random_input_len=1024,
            random_output_len=1024,
            random_range_ratio=0.0,
            image_count=1,
            image_resolution="1080p",
            random_image_count=False,
            image_format="jpeg",
            image_content="random",
            # traffic
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            seed=1,
            # output
            output_file=args.output_file,
            output_details=False,
            disable_tqdm=False,
            # features
            disable_stream=args.disable_stream,
            disable_ignore_eos=False,
            return_logprob=False,
            return_routed_experts=False,
            header=None,
            warmup_requests=args.warmup_requests,
            plot_throughput=False,
            apply_chat_template=False,
            tokenize_prompt=False,
            extra_request_body=None,
            flush_cache=False,
            print_requests=False,
            prompt_suffix="",
            # profiling (all off)
            profile=False,
            profile_activities=["CPU", "GPU"],
            profile_num_steps=None,
            profile_by_stage=False,
            profile_stages=None,
            profile_prefill_url=None,
            profile_decode_url=None,
            # LoRA (disabled)
            lora_name=None,
            lora_request_distribution="uniform",
            lora_zipf_alpha=1.5,
            # mooncake / trace
            use_trace_timestamps=False,
            mooncake_slowdown_factor=1.0,
            mooncake_num_rounds=1,
            pd_separated=False,
        )

        run_benchmark(bench_args)

    finally:
        if process is not None:
            print("[LAUNCHER] Shutting down server...")
            kill_process_tree(process.pid)
            time.sleep(3)
            print("[LAUNCHER] Server shutdown complete.")


if __name__ == "__main__":
    main()
