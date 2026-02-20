#!/usr/bin/env python3
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
jsonl = os.path.join(script_dir, "tmp", "gdn_decode_qk8_v16_d128_k_last.jsonl")
prefix = "gdn_decode_qk8_v16_d128_k_last"

if not os.path.isfile(jsonl):
    print(f"Error: {jsonl} not found", file=sys.stderr)
    sys.exit(1)

with open(jsonl) as f:
    lines = f.readlines()

deleted = 0
for line in lines[63:]:  # line 64 onward (0-based index 63)
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        uuid = obj.get("workload", {}).get("uuid")
        if uuid:
            path = os.path.join(script_dir, "tmp", f"{prefix}_{uuid}.safetensors")
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleted: {path}")
                deleted += 1
            else:
                print(f"Not found (skip): {path}")
    except Exception as e:
        print(f"Skip line: {e}", file=sys.stderr)

print(f"Done. Deleted {deleted} file(s).")
