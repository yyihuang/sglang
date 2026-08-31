"""Default-off internal hook for the sealed full-vocabulary KL evidence sink."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    KL_NORMALIZATION_ATOL,
    KL_SAMPLE_COUNT,
    KL_TOKEN_ID_ORDER,
    KL_VOCAB_CHUNK_SIZE,
)

AUTHORITY_ENV = "SGLANG_GDN_QUALIFICATION_KL_SINK_AUTHORITY"
AUTHORITY_SHA256_ENV = "SGLANG_GDN_QUALIFICATION_KL_SINK_AUTHORITY_SHA256"
KL_SINK_AUTHORITY_SCHEMA = "gdn-full-vocabulary-kl-sink-authority-v1"
KL_SINK_SAMPLE_SCHEMA = "gdn-full-vocabulary-kl-sink-sample-v1"
KL_SINK_RESPONSE_KEY = "gdn_full_vocabulary_kl_sink"


def sink_enabled() -> bool:
    return bool(os.environ.get(AUTHORITY_ENV) or os.environ.get(AUTHORITY_SHA256_ENV))


def marker_for_sample(vocab_size: int, sample_index: int) -> list[int]:
    if not 0 <= sample_index < KL_SAMPLE_COUNT:
        raise ValueError("KL sink sample index must be in [0, 48)")
    return [vocab_size - 1, vocab_size - 2, vocab_size - 1, vocab_size - 2, sample_index]


def _sample_from_marker(token_ids: object, vocab_size: int) -> int | None:
    if not isinstance(token_ids, list) or len(token_ids) != 5:
        return None
    sample_index = token_ids[-1]
    if type(sample_index) is not int or not 0 <= sample_index < KL_SAMPLE_COUNT:
        return None
    return sample_index if token_ids == marker_for_sample(vocab_size, sample_index) else None


@lru_cache(maxsize=1)
def load_authority() -> tuple[dict, str]:
    path_text = os.environ.get(AUTHORITY_ENV)
    expected_sha = os.environ.get(AUTHORITY_SHA256_ENV)
    if not path_text or not expected_sha:
        raise RuntimeError("KL sink authority environment is incomplete")
    path = Path(path_text)
    if not path.is_absolute() or path.name != "authority.json" or not path.is_file():
        raise RuntimeError("KL sink authority path is invalid")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError("KL sink authority SHA256 mismatch")
    authority = json.loads(raw)
    expected = {
        "schema": KL_SINK_AUTHORITY_SCHEMA,
        "sample_count": KL_SAMPLE_COUNT,
        "position_count": 512,
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "normalization_atol": KL_NORMALIZATION_ATOL,
    }
    if not isinstance(authority, dict) or any(authority.get(k) != v for k, v in expected.items()):
        raise RuntimeError("KL sink authority contract differs")
    root = Path(authority.get("root", ""))
    if root != path.parent or not root.is_absolute():
        raise RuntimeError("KL sink authority root differs")
    if authority.get("arm") not in {"baseline", "candidate"}:
        raise RuntimeError("KL sink authority arm differs")
    vocab_size = authority.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 2:
        raise RuntimeError("KL sink authority vocabulary differs")
    return authority, actual_sha


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write to KL sink")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def maybe_sink_full_vocab(
    logprobs,
    token_ids_logprobs: list,
    pruned_lens: list[int],
    split_pruned_len: int,
    log_normalizer,
) -> None:
    """Write one exact sample on TP rank zero when the sealed marker is present."""

    if not sink_enabled():
        return
    authority, authority_sha = load_authority()
    vocab_size = authority["vocab_size"]
    markers = [
        (index, _sample_from_marker(token_ids, vocab_size))
        for index, token_ids in enumerate(token_ids_logprobs)
    ]
    markers = [(index, sample) for index, sample in markers if sample is not None]
    if not markers:
        return
    if len(markers) != 1 or len(token_ids_logprobs) != 1 or markers[0][0] != 0:
        raise RuntimeError("KL sink requests must be isolated one sample per batch")
    if split_pruned_len != 0 or pruned_lens != [authority["position_count"] + 1]:
        raise RuntimeError("KL sink scoring-position layout differs")
    if tuple(logprobs.shape) != (
        authority["position_count"] + 1,
        vocab_size,
    ):
        raise RuntimeError("KL sink full-vocabulary logits shape differs")

    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return

    sample_index = markers[0][1]
    root = Path(authority["root"])
    shards = []
    position_count = authority["position_count"]
    for token_start in range(0, vocab_size, KL_VOCAB_CHUNK_SIZE):
        token_end = min(token_start + KL_VOCAB_CHUNK_SIZE, vocab_size)
        values = logprobs[:position_count, token_start:token_end].float()
        if log_normalizer is not None:
            row_max, row_log_sum = log_normalizer
            values = (values - row_max[:position_count, None]) - row_log_sum[:position_count, None]
        if not bool(values.isfinite().all().item()):
            raise RuntimeError("KL sink encountered non-finite logprobs")
        payload = values.detach().to(device="cpu").contiguous().numpy().astype("<f4", copy=False).tobytes()
        filename = f"sample-{sample_index:03d}-vocab-{token_start:06d}-{token_end:06d}.f32le"
        path = root / "shards" / filename
        _write_exclusive(path, payload)
        shards.append(
            {
                "path": f"shards/{filename}",
                "token_start": token_start,
                "token_end": token_end,
                "shape": [position_count, token_end - token_start],
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    receipt = {
        "schema": KL_SINK_SAMPLE_SCHEMA,
        "authority_sha256": authority_sha,
        "arm": authority["arm"],
        "sample_index": sample_index,
        "position_count": position_count,
        "vocab_size": vocab_size,
        "position_mapping": "first_512_rows_after_prompt_predecessor",
        "token_id_order": KL_TOKEN_ID_ORDER,
        "dtype": "float32",
        "byte_order": "little",
        "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
        "shards": shards,
    }
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _write_exclusive(root / "receipts" / f"sample-{sample_index:03d}.json", encoded)


def attach_sink_receipt(meta_info: dict, token_ids_logprob: object) -> None:
    if not sink_enabled():
        return
    authority, authority_sha = load_authority()
    sample_index = _sample_from_marker(token_ids_logprob, authority["vocab_size"])
    if sample_index is None:
        return
    path = Path(authority["root"]) / "receipts" / f"sample-{sample_index:03d}.json"
    if not path.is_file():
        raise RuntimeError("KL sink sample receipt is missing")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (
        receipt.get("authority_sha256") != authority_sha
        or receipt.get("arm") != authority["arm"]
        or receipt.get("sample_index") != sample_index
    ):
        raise RuntimeError("KL sink sample receipt identity differs")
    meta_info.pop("input_token_ids_logprobs", None)
    meta_info.pop("output_token_ids_logprobs", None)
    meta_info[KL_SINK_RESPONSE_KEY] = {
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }
