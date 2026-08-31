#!/usr/bin/env python3
"""Produce the exact final SGLang route artifact from authenticated sources."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess
from collections.abc import Mapping

from tools.gdn_public_qualification.contract import (
    EXACT_T4_ROUTE,
    ROUTE_ARTIFACT_SCHEMA,
    SGLANG_ROUTE_CONTRACT_ROWS,
)

CORE_SCHEMA = "flashinfer-gdn-noncp-decode-standalone-export-v1"
OVERLAY_SCHEMA = "flashinfer-gdn-noncp-public-overlay-v1"
CAKE_EXPORTER = pathlib.PurePosixPath("tools/export_flashinfer_gdn_noncp_decode.py")
ARCHITECTURES = ("sm_100a", "sm_103a")
REQUIRED_FLASHINFER_ROUTE_SOURCES = {
    "flashinfer/gdn_decode.py",
    "flashinfer/gdn_prefill.py",
    "flashinfer/jit/gdn_noncp.py",
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
ROUTE_RE = {
    "prefill": re.compile(r"flashinfer\.gdn_prefill\.noncp\.[a-z0-9_.-]+"),
    "decode": re.compile(r"flashinfer\.gdn_decode\.noncp\.[a-z0-9_.-]+"),
}


class RouteArtifactError(ValueError):
    """An input cannot prove the exact final route set."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RouteArtifactError(message)


def _full_hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    require(
        isinstance(value, str)
        and pattern.fullmatch(value) is not None
        and len(set(value)) > 1,
        f"{label} must be a resolved full lowercase hex identity",
    )
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: pathlib.Path, label: str) -> dict[str, object]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RouteArtifactError(f"{label} is not valid UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _git(root: pathlib.Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RouteArtifactError(f"Git validation failed for {root}") from error


def _validate_checkout(
    root: pathlib.Path, commit: str, tree: str, label: str
) -> None:
    require(root.is_absolute() and root == root.resolve(), f"{label} root must be normalized and absolute")
    require(root.is_dir(), f"{label} root is not a directory")
    require(_git(root, "rev-parse", "--is-inside-work-tree") == "true", f"{label} root is not a Git checkout")
    require(_git(root, "rev-parse", "HEAD^{commit}") == commit, f"{label} commit mismatch")
    require(_git(root, "rev-parse", "HEAD^{tree}") == tree, f"{label} tree mismatch")
    require(
        not _git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        f"{label} checkout is dirty",
    )


def _safe_checkout_file(root: pathlib.Path, relative: object, label: str) -> pathlib.Path:
    require(isinstance(relative, str), f"{label} path must be a string")
    pure = pathlib.PurePosixPath(relative)
    require(
        relative == pure.as_posix()
        and not pure.is_absolute()
        and pure.parts
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{label} path is not a normalized repository-relative path",
    )
    path = root.joinpath(*pure.parts)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular checkout file: {relative}")
    require(path.resolve().is_relative_to(root), f"{label} escapes its checkout")
    return path


def _validate_overlay(
    flashinfer_root: pathlib.Path,
    overlay: Mapping[str, object],
    core_sha256: str,
) -> dict[str, str]:
    require(overlay.get("schema") == OVERLAY_SCHEMA, "overlay manifest schema mismatch")
    require(overlay.get("core_manifest_sha256") == core_sha256, "overlay does not bind the core manifest")
    outputs = overlay.get("outputs")
    require(isinstance(outputs, Mapping) and bool(outputs), "overlay outputs must be a nonempty object")
    require(REQUIRED_FLASHINFER_ROUTE_SOURCES <= set(outputs), "overlay is missing required FlashInfer route sources")
    verified: dict[str, str] = {}
    for relative in sorted(outputs):
        record = outputs[relative]
        require(isinstance(record, Mapping) and set(record) == {"sha256", "size_bytes"}, f"overlay output record drift: {relative}")
        expected = _full_hex(record.get("sha256"), HEX64, f"overlay output SHA256: {relative}")
        path = _safe_checkout_file(flashinfer_root, relative, "overlay output")
        require(path.stat().st_size == record.get("size_bytes"), f"overlay output size mismatch: {relative}")
        require(_sha256(path) == expected, f"overlay output SHA256 mismatch: {relative}")
        verified[relative] = expected
    return verified


def _loader_literals(flashinfer_root: pathlib.Path) -> set[str]:
    relative = "flashinfer/jit/gdn_noncp.py"
    path = _safe_checkout_file(flashinfer_root, relative, "FlashInfer route loader")
    try:
        tree = ast.parse(path.read_text(), filename=relative)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise RouteArtifactError("authenticated FlashInfer route loader is not valid Python") from error
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _require_loader_literal(kind: str, route: str, literals: set[str]) -> None:
    if kind == "decode":
        require(route in literals, f"decode route lacks an authenticated loader literal: {route}")
        return
    base, separator, suffix = route.rpartition(".")
    require(
        separator == "."
        and suffix in {"dvsplit", "full_dv"}
        and base in literals
        and suffix in literals,
        f"prefill route lacks authenticated loader base/suffix literals: {route}",
    )


def _neutral_route(kind: str, route: object) -> str:
    require(isinstance(route, str), f"{kind} route ID must be a string")
    if kind == "prefill":
        result = route
    else:
        prefix = "flashinfer.gdn_decode."
        require(route.startswith(prefix) and not route.startswith(prefix + "noncp."), f"decode export route is not canonical: {route!r}")
        result = "flashinfer.gdn_decode.noncp." + route[len(prefix) :]
    require(ROUTE_RE[kind].fullmatch(result) is not None, f"invalid {kind} non-CP route: {result!r}")
    return result


def _derive_routes(
    core: Mapping[str, object], loader_literals: set[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    require(core.get("schema") == CORE_SCHEMA, "core manifest schema mismatch")
    require(core.get("architectures") == list(ARCHITECTURES), "core manifest architecture set or order mismatch")
    require(core.get("source_only") is True and core.get("binary_artifacts") is False, "core manifest is not source-only")
    require(core.get("manifest_only") is False, "manifest-only export cannot prove installed route sources")
    rows = core.get("contract_rows")
    require(isinstance(rows, list), "core manifest contract_rows must be a list")
    expected_keys = {
        (kind, contract_name, label)
        for kind, selectors in SGLANG_ROUTE_CONTRACT_ROWS.items()
        for contract_name, label in selectors
    }
    selected: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "core manifest contract row must be an object")
        key = (row.get("domain"), row.get("contract"), row.get("label"))
        if key in expected_keys:
            require(key not in selected, f"duplicate selected contract row: {key}")
            selected[key] = row
    require(set(selected) == expected_keys, "core manifest is missing a pinned SGLang route contract row")

    routes: dict[str, list[str]] = {"prefill": [], "decode": []}
    raw_routes_by_kind: dict[str, list[str]] = {"prefill": [], "decode": []}
    for key in sorted(selected):
        kind = key[0]
        dispatch = selected[key].get("dispatch")
        require(isinstance(dispatch, Mapping) and set(dispatch) == set(ARCHITECTURES), f"selected contract row architecture drift: {key}")
        raw_by_arch = []
        for arch in ARCHITECTURES:
            record = dispatch[arch]
            require(isinstance(record, Mapping), f"selected dispatch record is not an object: {key} {arch}")
            require(record.get("status") == "optimized", f"selected route is not optimized: {key} {arch}")
            raw_by_arch.append(record.get("route_id"))
        require(raw_by_arch[0] == raw_by_arch[1], f"selected route differs across architectures: {key}")
        raw_route = raw_by_arch[0]
        require(isinstance(raw_route, str), f"selected route ID is not a string: {key}")
        _require_loader_literal(kind, raw_route, loader_literals)
        routes[kind].append(_neutral_route(kind, raw_route))
        raw_routes_by_kind[kind].append(raw_route)
    return (
        {kind: sorted(set(values)) for kind, values in routes.items()},
        {kind: sorted(set(values)) for kind, values in raw_routes_by_kind.items()},
    )


def produce(
    *,
    cake_root: pathlib.Path,
    cake_commit: str,
    cake_tree: str,
    cake_exporter_sha256: str,
    flashinfer_root: pathlib.Path,
    flashinfer_commit: str,
    flashinfer_tree: str,
    core_manifest: pathlib.Path,
    core_manifest_sha256: str,
    overlay_manifest: pathlib.Path,
    overlay_manifest_sha256: str,
) -> dict[str, object]:
    cake_commit = _full_hex(cake_commit, HEX40, "Cake commit")
    cake_tree = _full_hex(cake_tree, HEX40, "Cake tree")
    flashinfer_commit = _full_hex(flashinfer_commit, HEX40, "FlashInfer commit")
    flashinfer_tree = _full_hex(flashinfer_tree, HEX40, "FlashInfer tree")
    cake_exporter_sha256 = _full_hex(cake_exporter_sha256, HEX64, "Cake exporter SHA256")
    core_manifest_sha256 = _full_hex(core_manifest_sha256, HEX64, "core manifest SHA256")
    overlay_manifest_sha256 = _full_hex(overlay_manifest_sha256, HEX64, "overlay manifest SHA256")
    _validate_checkout(cake_root, cake_commit, cake_tree, "Cake")
    _validate_checkout(flashinfer_root, flashinfer_commit, flashinfer_tree, "FlashInfer")
    exporter = _safe_checkout_file(cake_root, CAKE_EXPORTER.as_posix(), "Cake exporter")
    require(_sha256(exporter) == cake_exporter_sha256, "Cake exporter SHA256 mismatch")
    core = _load_object(core_manifest, "core manifest")
    overlay = _load_object(overlay_manifest, "overlay manifest")
    require(_sha256(core_manifest) == core_manifest_sha256, "core manifest SHA256 mismatch")
    require(_sha256(overlay_manifest) == overlay_manifest_sha256, "overlay manifest SHA256 mismatch")
    verified_outputs = _validate_overlay(flashinfer_root, overlay, core_manifest_sha256)
    routes, raw_routes = _derive_routes(core, _loader_literals(flashinfer_root))
    require(EXACT_T4_ROUTE in routes["decode"], "pinned exact T=4 decode route is absent")
    return {
        "schema": ROUTE_ARTIFACT_SCHEMA,
        "status": "PASS",
        "cake": {"commit": cake_commit, "tree": cake_tree},
        "flashinfer": {"commit": flashinfer_commit, "tree": flashinfer_tree},
        "route_sources": {
            "cake_exporter_sha256": cake_exporter_sha256,
            "core_manifest_sha256": core_manifest_sha256,
            "overlay_manifest_sha256": overlay_manifest_sha256,
            "flashinfer_outputs_sha256": verified_outputs,
            "selected_raw_routes": raw_routes,
        },
        "qualification_contract_rows": {
            kind: [
                {"contract": contract_name, "label": label}
                for contract_name, label in selectors
            ]
            for kind, selectors in SGLANG_ROUTE_CONTRACT_ROWS.items()
        },
        "expected_candidate_routes": routes,
        "exact_t4_route": EXACT_T4_ROUTE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cake-root", type=pathlib.Path, required=True)
    parser.add_argument("--cake-commit", required=True)
    parser.add_argument("--cake-tree", required=True)
    parser.add_argument("--cake-exporter-sha256", required=True)
    parser.add_argument("--flashinfer-root", type=pathlib.Path, required=True)
    parser.add_argument("--flashinfer-commit", required=True)
    parser.add_argument("--flashinfer-tree", required=True)
    parser.add_argument("--core-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--core-manifest-sha256", required=True)
    parser.add_argument("--overlay-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        require(args.output.is_absolute() and args.output == args.output.resolve(), "output must be normalized and absolute")
        require(args.output.parent.is_dir() and not args.output.exists(), "output must be a fresh file in an existing directory")
        artifact = produce(
            cake_root=args.cake_root,
            cake_commit=args.cake_commit,
            cake_tree=args.cake_tree,
            cake_exporter_sha256=args.cake_exporter_sha256,
            flashinfer_root=args.flashinfer_root,
            flashinfer_commit=args.flashinfer_commit,
            flashinfer_tree=args.flashinfer_tree,
            core_manifest=args.core_manifest,
            core_manifest_sha256=args.core_manifest_sha256,
            overlay_manifest=args.overlay_manifest,
            overlay_manifest_sha256=args.overlay_manifest_sha256,
        )
        with args.output.open("x") as stream:
            stream.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": "PASS"}, sort_keys=True))
        return 0
    except RouteArtifactError as error:
        parser.exit(78, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
