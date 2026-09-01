"""Audit a raw final GDN qualification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.gdn_public_qualification.contract import (
    QualificationError,
    audit_receipt,
    canonical_json_text,
    load_strict_json,
)


def _canonical_json(value: object) -> bytes:
    return (canonical_json_text(value) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.receipt.read_bytes()
    try:
        receipt = load_strict_json(raw.decode())
        audit = audit_receipt(receipt, args.receipt.resolve().parent)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, QualificationError) as exc:
        parser.error(str(exc))
    audit["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    encoded = _canonical_json(audit)
    args.output.write_bytes(encoded)
    print(json.dumps(audit, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
