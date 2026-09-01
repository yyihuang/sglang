"""Default-off HTTP-ingress authority for exact-once GSM8K evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    ACCURACY_SERVER_AUTHORITY_SCHEMA,
    ACCURACY_SERVER_RECEIPT_SCHEMA,
    PROMPT_COUNT,
    canonical_json_sha256,
    canonical_json_text,
    load_strict_json,
)

AUTHORITY_ENV = "SGLANG_GDN_QUALIFICATION_ACCURACY_AUTHORITY"
AUTHORITY_SHA256_ENV = "SGLANG_GDN_QUALIFICATION_ACCURACY_AUTHORITY_SHA256"
AUTHORITY_SCHEMA = ACCURACY_SERVER_AUTHORITY_SCHEMA
RECEIPT_SCHEMA = ACCURACY_SERVER_RECEIPT_SCHEMA

_REQUEST_RE = re.compile(
    r"^gdn-gsm8k-([0-9a-f]{64})-(baseline|candidate)-([0-9]{4})$"
)
_LOCK = threading.Lock()


def hook_enabled() -> bool:
    return bool(os.environ.get(AUTHORITY_ENV) or os.environ.get(AUTHORITY_SHA256_ENV))


def _write_exclusive(path: Path, value: object) -> None:
    payload = (canonical_json_text(value) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write to accuracy request authority")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@lru_cache(maxsize=1)
def load_authority() -> tuple[dict, str, Path]:
    path_text = os.environ.get(AUTHORITY_ENV)
    expected_sha = os.environ.get(AUTHORITY_SHA256_ENV)
    if not path_text or not expected_sha:
        raise RuntimeError("accuracy request authority environment is incomplete")
    path = Path(path_text)
    if not path.is_absolute() or path.name != "authority.json" or not path.is_file():
        raise RuntimeError("accuracy request authority path is invalid")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError("accuracy request authority SHA256 mismatch")
    try:
        authority = load_strict_json(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"accuracy request authority is unreadable: {exc}") from exc
    if not isinstance(authority, dict):
        raise RuntimeError("accuracy request authority must be an object")
    expected = {
        "schema": AUTHORITY_SCHEMA,
        "prompt_count": PROMPT_COUNT,
        "accepted_count_per_prompt": 1,
        "completed_count_per_prompt": 1,
    }
    if any(authority.get(key) != value for key, value in expected.items()):
        raise RuntimeError("accuracy request authority contract differs")
    if authority.get("arm") not in {"baseline", "candidate"}:
        raise RuntimeError("accuracy request authority arm differs")
    root = Path(authority.get("root", ""))
    if not root.is_absolute() or root != path.parent:
        raise RuntimeError("accuracy request authority root differs")
    return authority, actual_sha, root


def begin_request(obj) -> dict | None:
    """Atomically admit one qualification rid at the HTTP ingress boundary."""

    if not hook_enabled():
        return None
    rid = getattr(obj, "rid", None)
    if not isinstance(rid, str):
        return None
    match = _REQUEST_RE.fullmatch(rid)
    if match is None:
        return None
    campaign_id, arm, index_text = match.groups()
    index = int(index_text)
    authority, authority_sha, root = load_authority()
    if arm != authority["arm"] or not 0 <= index < PROMPT_COUNT:
        raise ValueError("qualification GSM8K request identity differs from server authority")
    if bool(getattr(obj, "stream", False)):
        raise ValueError("qualification GSM8K requests must be non-streaming")
    with _LOCK:
        campaign_path = root / "campaign.json"
        campaign = {
            "schema": "gdn-gsm8k-server-campaign-v1",
            "authority_sha256": authority_sha,
            "arm": arm,
            "campaign_id": campaign_id,
        }
        if campaign_path.exists():
            try:
                observed_campaign = load_strict_json(campaign_path.read_text())
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"qualification campaign authority is unreadable: {exc}") from exc
            if observed_campaign != campaign:
                raise ValueError("qualification server is already bound to a different campaign")
        else:
            _write_exclusive(campaign_path, campaign)
        accepted = {
            "schema": "gdn-gsm8k-server-request-accepted-v1",
            "authority_sha256": authority_sha,
            "arm": arm,
            "campaign_id": campaign_id,
            "question_index": index,
            "request_id": rid,
        }
        try:
            _write_exclusive(root / "accepted" / f"prompt-{index:04d}.json", accepted)
        except FileExistsError as exc:
            raise ValueError(
                f"duplicate qualification GSM8K request for prompt {index}"
            ) from exc
    return accepted


def finish_request(token: dict | None, response: object) -> None:
    """Seal completion and publish the aggregate receipt after prompt 1,314."""

    if token is None:
        return
    authority, authority_sha, root = load_authority()
    completed = {
        "schema": "gdn-gsm8k-server-request-completed-v1",
        "authority_sha256": authority_sha,
        "arm": token["arm"],
        "campaign_id": token["campaign_id"],
        "question_index": token["question_index"],
        "request_id": token["request_id"],
        "response_sha256": canonical_json_sha256(response),
    }
    with _LOCK:
        try:
            _write_exclusive(
                root / "completed" / f"prompt-{token['question_index']:04d}.json",
                completed,
            )
        except FileExistsError as exc:
            raise ValueError(
                f"duplicate qualification GSM8K completion for prompt {token['question_index']}"
            ) from exc
        completed_paths = sorted((root / "completed").glob("prompt-*.json"))
        if len(completed_paths) != PROMPT_COUNT:
            return
        records = []
        for index in range(PROMPT_COUNT):
            accepted_path = root / "accepted" / f"prompt-{index:04d}.json"
            completed_path = root / "completed" / f"prompt-{index:04d}.json"
            if not accepted_path.is_file() or not completed_path.is_file():
                raise ValueError("qualification server request evidence is incomplete")
            accepted = load_strict_json(accepted_path.read_text())
            observed_completed = load_strict_json(completed_path.read_text())
            records.append(
                {
                    "question_index": index,
                    "request_id": accepted.get("request_id"),
                    "accepted_count": 1,
                    "completed_count": 1,
                    "response_sha256": observed_completed.get("response_sha256"),
                }
            )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "authority_sha256": authority_sha,
            "arm": authority["arm"],
            "campaign_id": token["campaign_id"],
            "prompt_count": PROMPT_COUNT,
            "records": records,
        }
        _write_exclusive(root / "receipt.json", receipt)
