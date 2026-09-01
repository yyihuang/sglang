from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.gdn_public_qualification.assemble import assemble
from tools.gdn_public_qualification.contract import (
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXACT_T4_ROUTE,
    HASHES,
    PLAN_SCHEMA,
    ROUTE_ARTIFACT_SCHEMA,
    SCHEMA,
    SGLANG_ROUTE_CONTRACT_ROWS,
    SOURCE_COMMIT,
    SOURCE_TREE,
    WORKLOADS,
    QualificationError,
    expected_provenance,
)


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_artifact(expected_routes: dict[str, list[str]]) -> dict[str, object]:
    return {
        "schema": ROUTE_ARTIFACT_SCHEMA,
        "status": "PASS",
        "cake": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
        "flashinfer": {
            "commit": expected_provenance()["flashinfer_commit"],
            "tree": expected_provenance()["flashinfer_tree"],
        },
        "route_sources": {
            "cake_exporter_sha256": HASHES["exporter_sha256"],
            "core_manifest_sha256": HASHES["core_manifest_sha256"],
            "overlay_manifest_sha256": HASHES["overlay_manifest_sha256"],
            "flashinfer_outputs_sha256": {
                "flashinfer/gdn_decode.py": "1" * 64,
                "flashinfer/gdn_prefill.py": "2" * 64,
                "flashinfer/jit/gdn_noncp.py": "3" * 64,
            },
            "selected_raw_routes": {
                "prefill": expected_routes["prefill"],
                "decode": [
                    route.replace(
                        "flashinfer.gdn_decode.noncp.", "flashinfer.gdn_decode."
                    )
                    for route in expected_routes["decode"]
                ],
            },
        },
        "qualification_contract_rows": {
            kind: [
                {"contract": contract_name, "label": label}
                for contract_name, label in selectors
            ]
            for kind, selectors in SGLANG_ROUTE_CONTRACT_ROWS.items()
        },
        "expected_candidate_routes": expected_routes,
        "exact_t4_route": EXACT_T4_ROUTE,
    }


class AssembleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.expected_routes = {
            "prefill": ["flashinfer.gdn_prefill.noncp.full_dv"],
            "decode": [EXACT_T4_ROUTE],
        }
        self.dataset = self.root / "gsm8k.jsonl"
        self.dataset.write_text("sealed\n")
        self.prompt_ids = self.root / "gsm8k-prompt-ids.json"
        self.prompt_ids.write_text("[]\n")
        self.route_artifact = self.root / "route-artifact.json"
        self.route_sha256 = _write(
            self.route_artifact, _route_artifact(self.expected_routes)
        )
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text("{}\n")
        self.campaign_id = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        provenance = expected_provenance()
        provenance.update(
            {
                "qualification_commit": "a" * 40,
                "qualification_tree": "b" * 40,
                "compute_capability": [10, 3],
                "gpu_name": "NVIDIA GB300",
                "cuda_version": "13.0",
                "tp_size": 4,
                "tp_ranks": [0, 1, 2, 3],
            }
        )
        self.plan = {
            "schema": PLAN_SCHEMA,
            "provenance": provenance,
            "bindings": {
                "artifacts": {
                    "gsm8k_dataset": str(self.dataset),
                    "gsm8k_prompt_ids": str(self.prompt_ids),
                },
                "route_artifact": {
                    "path": str(self.route_artifact),
                    "sha256": self.route_sha256,
                },
            },
            "routes": {
                "artifact": {
                    "schema": ROUTE_ARTIFACT_SCHEMA,
                    "sha256": self.route_sha256,
                },
                "expected_candidate_routes": self.expected_routes,
            },
        }

        self.accuracy = {}
        for arm in ("baseline", "candidate"):
            path = self.root / f"accuracy.{arm}.json"
            self.accuracy[arm] = path
            _write(
                path,
                {
                    "arm": arm,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.campaign_id,
                    "dataset_sha256": HASHES["gsm8k_dataset_sha256"],
                    "prompt_ids_sha256": HASHES["gsm8k_prompt_ids_sha256"],
                    "request_payload": "input_ids",
                    "sampling_params": {},
                    "server_config": {},
                    "model_identity": {},
                    "request_ledger": {},
                    "server_request_evidence": {},
                    "score": 1.0,
                    "prompts": [],
                },
            )
        self.mtp = {}
        self.routes = {}
        for arm in ("baseline", "candidate"):
            mtp = self.root / f"mtp.{arm}.json"
            self.mtp[arm] = mtp
            _write(
                mtp,
                {
                    "arm": arm,
                    "input_ids_sha256": HASHES["longbench_first48_ids_sha256"],
                    "prompt_index": 0,
                    "request_count": 1,
                    "sampling_params": {},
                    "server_config": {},
                    "output_ids": [1],
                    "output_ids_sha256": "4" * 64,
                    "measured_runtime_seconds": 1.0,
                },
            )
            routes = self.root / f"routes.{arm}.json"
            self.routes[arm] = routes
            _write(routes, {"arm": arm, "ranks": []})
        self.kl = {}
        for arm in ("baseline", "candidate"):
            path = self.root / f"kl.{arm}.json"
            self.kl[arm] = path
            _write(path, {"arm": arm})

        self.performance = self.root / "performance"
        self.performance.mkdir()
        for workload_id, input_ids_sha256 in WORKLOADS.items():
            for sequence_index, arm in enumerate(ABBA_ORDER):
                _write(
                    self.performance
                    / f"performance.{workload_id}.{sequence_index:02d}.{arm}.json",
                    {
                        "arm": arm,
                        "workload_id": workload_id,
                        "input_ids_sha256": input_ids_sha256,
                        "throughput_tokens_per_second": 100.0,
                        "measured_runtime_seconds": 1.0,
                        "output_tokens": 512,
                    },
                )
        self.args = argparse.Namespace(
            plan=self.plan_path,
            accuracy_baseline=self.accuracy["baseline"],
            accuracy_candidate=self.accuracy["candidate"],
            mtp_baseline=self.mtp["baseline"],
            mtp_candidate=self.mtp["candidate"],
            routes_baseline=self.routes["baseline"],
            routes_candidate=self.routes["candidate"],
            kl_baseline_manifest=self.kl["baseline"],
            kl_candidate_manifest=self.kl["candidate"],
            performance_dir=self.performance,
            physical_start_ns=1_000_000_000,
            measured_start_ns=2_000_000_000,
            finish_ns=4_000_000_000,
            output=self.root / "result.json",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_collector_set_assembles_one_v2_result(self) -> None:
        with (
            patch(
                "tools.gdn_public_qualification.assemble.load_campaign_plan",
                return_value=self.plan,
            ),
            patch(
                "tools.gdn_public_qualification.assemble.recompute_kl_summary",
                return_value={
                    "mean_sample_kl": 0.001,
                    "max_sample_kl": 0.002,
                    "max_position_kl": 0.003,
                },
            ),
        ):
            result = assemble(self.args)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["accuracy"]["campaign_id"], self.campaign_id)
        self.assertEqual(result["routes"]["artifact"]["sha256"], self.route_sha256)
        self.assertEqual(
            result["routes"]["expected_candidate_routes"], self.expected_routes
        )
        self.assertEqual(result["performance"]["bootstrap_samples"], BOOTSTRAP_SAMPLES)
        self.assertEqual(result["performance"]["bootstrap_seed"], BOOTSTRAP_SEED)
        self.assertEqual(len(result["performance"]["workloads"]), 2)
        self.assertTrue(
            all(
                len(workload["observations"]) == 16
                for workload in result["performance"]["workloads"]
            )
        )

    def test_duplicate_or_nonfinite_collector_json_fails_closed(self) -> None:
        self.mtp["baseline"].write_text('{"arm":"baseline","arm":"candidate"}\n')
        with patch(
            "tools.gdn_public_qualification.assemble.load_campaign_plan",
            return_value=self.plan,
        ):
            with self.assertRaisesRegex(QualificationError, "duplicate JSON object key"):
                assemble(self.args)

        self.mtp["baseline"].write_text('{"arm":"baseline","value":NaN}\n')
        with patch(
            "tools.gdn_public_qualification.assemble.load_campaign_plan",
            return_value=self.plan,
        ):
            with self.assertRaisesRegex(QualificationError, "non-finite JSON constant"):
                assemble(self.args)

    def test_route_artifact_hash_must_match_the_rendered_plan(self) -> None:
        self.plan["bindings"]["route_artifact"]["sha256"] = "0" * 64
        with patch(
            "tools.gdn_public_qualification.assemble.load_campaign_plan",
            return_value=self.plan,
        ):
            with self.assertRaisesRegex(
                QualificationError, "route artifact hash differs from the campaign plan"
            ):
                assemble(self.args)


if __name__ == "__main__":
    unittest.main()
