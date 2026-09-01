from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from tools.gdn_public_qualification.contract import (
    EXACT_T4_ROUTE,
    HASHES,
    QualificationError,
    validate_route_artifact,
)
from tools.gdn_public_qualification.produce_route_artifact import (
    CORE_SCHEMA,
    OVERLAY_SCHEMA,
    RouteArtifactError,
    produce,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: pathlib.Path, value: object) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return _sha(data)


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _repo(root: pathlib.Path, files: dict[str, bytes]) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(root, "add", "--all")
    _git(root, "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD^{commit}"), _git(root, "rev-parse", "HEAD^{tree}")


def _dispatch(route: str) -> dict[str, dict[str, str]]:
    return {arch: {"status": "optimized", "route_id": route} for arch in ("sm_100a", "sm_103a")}


def _core() -> dict[str, object]:
    rows = (
        ("prefill", "prefill_focus", "correctness_sglang_tp4_bf16_indexed_b5_s64", "flashinfer.gdn_prefill.noncp.dvsplit"),
        ("prefill", "prefill_focus", "correctness_sglang_tp4_bf16_indexed_checkpoint_b7_t421", "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"),
        ("decode", "decode_bf16_serving", "bf16_sglang_qwen_tp4_decode_t1", "flashinfer.gdn_decode.indexed_bf16_t1.tile16_fullwarp"),
        ("decode", "decode_bf16_serving", "bf16_sglang_qwen_tp4_verify_t4", "flashinfer.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp"),
    )
    return {
        "schema": CORE_SCHEMA,
        "architectures": ["sm_100a", "sm_103a"],
        "source_only": True,
        "binary_artifacts": False,
        "manifest_only": False,
        "contract_rows": [
            {"domain": domain, "contract": contract, "label": label, "dispatch": _dispatch(route)}
            for domain, contract, label, route in reversed(rows)
        ],
    }


class RouteArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temp.name).resolve()
        self.exporter = b"# exporter\n"
        self.cake = root / "cake"
        self.cake_commit, self.cake_tree = _repo(
            self.cake, {"tools/export_flashinfer_gdn_noncp_decode.py": self.exporter}
        )
        loader = b'''ROUTES = [
"flashinfer.gdn_prefill.noncp", "flashinfer.gdn_prefill.noncp.checkpoints",
"dvsplit", "full_dv",
"flashinfer.gdn_decode.indexed_bf16_t1.tile16_fullwarp",
"flashinfer.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp",
]\n'''
        self.fi_files = {
            "flashinfer/gdn_prefill.py": b"# prefill\n",
            "flashinfer/gdn_decode.py": b"# decode\n",
            "flashinfer/jit/gdn_noncp.py": loader,
        }
        self.fi = root / "flashinfer"
        self.fi_commit, self.fi_tree = _repo(self.fi, self.fi_files)
        self.core_path = root / "core.json"
        self.core_sha = _json(self.core_path, _core())
        self.overlay = {
            "schema": OVERLAY_SCHEMA,
            "core_manifest_sha256": self.core_sha,
            "outputs": {
                name: {"sha256": _sha(data), "size_bytes": len(data)}
                for name, data in self.fi_files.items()
            },
        }
        self.overlay_path = root / "overlay.json"
        self.overlay_sha = _json(self.overlay_path, self.overlay)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> dict[str, object]:
        return {
            "cake_root": self.cake, "cake_commit": self.cake_commit,
            "cake_tree": self.cake_tree, "cake_exporter_sha256": _sha(self.exporter),
            "flashinfer_root": self.fi, "flashinfer_commit": self.fi_commit,
            "flashinfer_tree": self.fi_tree, "core_manifest": self.core_path,
            "core_manifest_sha256": self.core_sha,
            "overlay_manifest": self.overlay_path,
            "overlay_manifest_sha256": self.overlay_sha,
        }

    def consumer_pins(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch(
                "tools.gdn_public_qualification.contract.SOURCE_COMMIT",
                self.cake_commit,
            )
        )
        stack.enter_context(
            patch(
                "tools.gdn_public_qualification.contract.SOURCE_TREE",
                self.cake_tree,
            )
        )
        stack.enter_context(
            patch(
                "tools.gdn_public_qualification.contract.FLASHINFER_COMMIT",
                self.fi_commit,
            )
        )
        stack.enter_context(
            patch(
                "tools.gdn_public_qualification.contract.FLASHINFER_TREE",
                self.fi_tree,
            )
        )
        stack.enter_context(
            patch.dict(
                HASHES,
                {
                    "exporter_sha256": _sha(self.exporter),
                    "core_manifest_sha256": self.core_sha,
                    "overlay_manifest_sha256": self.overlay_sha,
                },
                clear=False,
            )
        )
        return stack

    def test_deterministic_exact_routes(self) -> None:
        first, second = produce(**self.args()), produce(**self.args())
        self.assertEqual(first, second)
        self.assertEqual(first["exact_t4_route"], EXACT_T4_ROUTE)
        self.assertEqual(
            first["expected_candidate_routes"]["decode"],
            ["flashinfer.gdn_decode.noncp.indexed_bf16_t1.tile16_fullwarp", EXACT_T4_ROUTE],
        )
        self.assertEqual(
            first["expected_candidate_routes"]["prefill"],
            ["flashinfer.gdn_prefill.noncp.checkpoints.dvsplit", "flashinfer.gdn_prefill.noncp.dvsplit"],
        )
        with self.consumer_pins():
            self.assertEqual(
                validate_route_artifact(first), first["expected_candidate_routes"]
            )

    def test_artifact_consumer_rejects_unbound_expected_routes(self) -> None:
        artifact = produce(**self.args())
        artifact["expected_candidate_routes"]["prefill"] = [
            "flashinfer.gdn_prefill.noncp.caller_supplied"
        ]
        with self.consumer_pins():
            with self.assertRaisesRegex(
                QualificationError,
                "expected routes differ from selected raw routes",
            ):
                validate_route_artifact(artifact)

    def test_identity_and_dirty_sources_fail_closed(self) -> None:
        args = self.args()
        args["cake_commit"] = self.cake_commit[:12]
        with self.assertRaisesRegex(RouteArtifactError, "resolved full"):
            produce(**args)
        dirty = self.cake / "untracked"
        dirty.write_text("dirty")
        with self.assertRaisesRegex(RouteArtifactError, "checkout is dirty"):
            produce(**self.args())
        dirty.unlink()

    def test_authenticated_loader_literal_is_required(self) -> None:
        core = copy.deepcopy(_core())
        row = next(row for row in core["contract_rows"] if row["label"] == "bf16_sglang_qwen_tp4_verify_t4")
        row["dispatch"] = _dispatch("flashinfer.gdn_decode.indexed_bf16_verify_t4.unproved")
        core_sha = _json(self.core_path, core)
        overlay = copy.deepcopy(self.overlay)
        overlay["core_manifest_sha256"] = core_sha
        overlay_sha = _json(self.overlay_path, overlay)
        args = self.args()
        args.update(core_manifest_sha256=core_sha, overlay_manifest_sha256=overlay_sha)
        with self.assertRaisesRegex(RouteArtifactError, "authenticated loader literal"):
            produce(**args)


if __name__ == "__main__":
    unittest.main()
