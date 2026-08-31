"""Launch SGLang with a sealed, qualification-only full-vocabulary KL sink."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    KL_NORMALIZATION_ATOL,
    KL_SAMPLE_COUNT,
    KL_TOKEN_ID_ORDER,
    KL_VOCAB_CHUNK_SIZE,
)
from tools.gdn_public_qualification.kl_sink_hook import (
    AUTHORITY_ENV,
    AUTHORITY_SHA256_ENV,
    KL_SINK_AUTHORITY_SCHEMA,
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def prepare_sink_root(root: Path, arm: str, vocab_size: int) -> tuple[Path, str]:
    if not root.is_absolute():
        raise ValueError("KL sink root must be absolute")
    if root.exists():
        raise ValueError(f"KL sink root already exists: {root}")
    parent = root.parent.resolve(strict=True)
    if not parent.is_dir() or root != parent / root.name:
        raise ValueError("KL sink root must be a direct, non-symlink child of an existing directory")
    if arm not in {"baseline", "candidate"}:
        raise ValueError("KL sink arm must be baseline or candidate")
    if type(vocab_size) is not int or vocab_size <= 2:
        raise ValueError("KL sink vocab size must be an integer > 2")

    root.mkdir(mode=0o700)
    (root / "shards").mkdir(mode=0o700)
    (root / "receipts").mkdir(mode=0o700)
    authority = {
        "schema": KL_SINK_AUTHORITY_SCHEMA,
        "root": str(root),
        "arm": arm,
        "vocab_size": vocab_size,
        "sample_count": KL_SAMPLE_COUNT,
        "position_count": 512,
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "normalization_atol": KL_NORMALIZATION_ATOL,
    }
    payload = _canonical_json(authority)
    authority_path = root / "authority.json"
    with authority_path.open("xb") as handle:
        handle.write(payload)
    authority_path.chmod(0o400)
    return authority_path, hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sink-root", type=Path, required=True)
    parser.add_argument("--sink-arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--sink-vocab-size", type=int, required=True)
    parser.add_argument("sglang_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.sglang_args[1:] if args.sglang_args[:1] == ["--"] else args.sglang_args
    if not server_args:
        parser.error("SGLang server arguments are required after --")
    authority_path, authority_sha256 = prepare_sink_root(
        args.sink_root, args.sink_arm, args.sink_vocab_size
    )
    os.environ[AUTHORITY_ENV] = str(authority_path)
    os.environ[AUTHORITY_SHA256_ENV] = authority_sha256
    sys.argv = ["sglang.launch_server", *server_args]
    runpy.run_module("sglang.launch_server", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
