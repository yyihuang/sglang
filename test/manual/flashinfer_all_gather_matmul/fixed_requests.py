#!/usr/bin/env python3
"""Run 32 deterministic GSM8K parity requests and retain token logprobs."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def one_example(rows, index, include_answer):
    text = f"Question: {rows[index]['question']}\nAnswer:"
    if include_answer:
        text += f" {rows[index]['answer']}"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.gsm8k.read_text().splitlines()]
    prefix = "".join(one_example(rows, i, True) + "\n\n" for i in range(5))
    prompts = [prefix + one_example(rows, i, False) for i in range(5, 37)]
    response = requests.post(
        args.base_url.rstrip("/") + "/generate",
        json={
            "text": prompts,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 32,
                "ignore_eos": False,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
        timeout=1800,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} responses, got {type(payload).__name__}"
        )

    normalized = []
    for index, (prompt, item) in enumerate(zip(prompts, payload)):
        output_logprobs = item["meta_info"]["output_token_logprobs"]
        output_ids = item.get("output_ids") or [entry[1] for entry in output_logprobs]
        normalized.append(
            {
                "index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_ids": output_ids,
                "output_logprobs": [entry[0] for entry in output_logprobs],
                "text": item.get("text", ""),
            }
        )
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

