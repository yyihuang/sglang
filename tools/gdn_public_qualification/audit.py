"""Audit a raw final GDN qualification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.gdn_public_qualification.contract import QualificationError, audit_receipt


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.receipt.read_bytes()
    receipt = json.loads(raw)
    try:
        audit = audit_receipt(receipt)
    except QualificationError as exc:
        parser.error(str(exc))
    audit["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    encoded = _canonical_json(audit)
    args.output.write_bytes(encoded)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
