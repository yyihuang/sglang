#!/usr/bin/env python3
"""Fail closed unless every TP rank exercised the expected experiment route."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("explicit", "candidate"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.artifact_dir.glob("agmm-route-rank*.json"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four rank summaries, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    for rank, row in enumerate(rows):
        if row["rank"] != rank or row["world_size"] != 4:
            raise RuntimeError(f"Unexpected TP metadata: {row}")
        if row["variant"] != args.variant:
            raise RuntimeError(f"Unexpected variant in rank {rank}: {row['variant']}")
        if row[f"{args.variant}_hits"] <= 0:
            raise RuntimeError(f"Rank {rank} did not hit {args.variant}")
        if row["unique_qkv_modules"] != 80 or row["validated_calls"] != 80:
            raise RuntimeError(f"Rank {rank} did not validate all 80 layers: {row}")
        if args.variant == "candidate":
            if row["candidate_backend"] != "cake":
                raise RuntimeError(f"Rank {rank} did not request Cake: {row}")
            if row["cake_backend_requests"] != row["candidate_hits"]:
                raise RuntimeError(f"Rank {rank} Cake request count mismatch: {row}")
    result = {"variant": args.variant, "rank_count": 4, "ranks": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

