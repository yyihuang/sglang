from __future__ import annotations

import copy
import math
import unittest

from tools.gdn_public_qualification.contract import (
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    PROMPT_COUNT,
    SCHEMA,
    WORKLOADS,
    QualificationError,
    audit_receipt,
    expected_provenance,
)


def _prompt_rows(correct_count: int) -> list[dict]:
    return [
        {
            "question_index": index,
            "source_row_index": index + 5,
            "request_count": 1,
            "correct": index < correct_count,
        }
        for index in range(PROMPT_COUNT)
    ]


def _accuracy_arm(correct_count: int) -> dict:
    return {
        "score": correct_count / PROMPT_COUNT,
        "prompts": _prompt_rows(correct_count),
    }


def _observations(candidate_ratio: float) -> list[dict]:
    rows = []
    for index, arm in enumerate(ABBA_ORDER):
        rows.append(
            {
                "sequence_index": index,
                "arm": arm,
                "throughput_tokens_per_second": 100.0 * (candidate_ratio if arm == "candidate" else 1.0),
                "measured_runtime_seconds": 1.0,
            }
        )
    return rows


def _receipt() -> dict:
    provenance = expected_provenance()
    provenance.update(
        {
            "qualification_commit": "a" * 40,
            "compute_capability": [10, 3],
            "gpu_name": "NVIDIA GB300",
            "cuda_version": "13.0",
            "tp_size": 4,
            "tp_ranks": [0, 1, 2, 3],
        }
    )
    expected_routes = {
        "prefill": ["cake.gdn_prefill.noncp.full_dv"],
        "decode": ["cake.gdn_decode.noncp.tile16_fullwarp"],
    }
    return {
        "schema": SCHEMA,
        "provenance": provenance,
        "campaign": {
            "started_at": "2026-08-30T00:00:00Z",
            "finished_at": "2026-08-30T01:00:00Z",
            "physical_turnaround_seconds": 3600.0,
            "measured_runtime_seconds": 3200.0,
        },
        "accuracy": {
            "arms": {
                "baseline": _accuracy_arm(1223),
                "candidate": _accuracy_arm(1223),
            },
            "kl": {
                "mean_kl": 0.0,
                "records": [
                    {
                        "sample_index": index,
                        "baseline_logprobs": [0.0, -1.0],
                        "candidate_logprobs": [0.0, -1.0],
                    }
                    for index in range(48)
                ],
            },
        },
        "routes": {
            "expected_candidate_routes": expected_routes,
            "arms": {
                "baseline": [
                    {
                        "rank": rank,
                        "prefill_routes": [],
                        "decode_routes": [],
                        "cake_route_count": 0,
                        "fallback_count": 0,
                        "route_error_count": 0,
                    }
                    for rank in range(4)
                ],
                "candidate": [
                    {
                        "rank": rank,
                        "prefill_routes": expected_routes["prefill"],
                        "decode_routes": expected_routes["decode"],
                        "cake_route_count": 2,
                        "fallback_count": 0,
                        "route_error_count": 0,
                    }
                    for rank in range(4)
                ],
            },
        },
        "performance": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "workloads": [
                {
                    "workload_id": workload_id,
                    "input_ids_sha256": digest,
                    "observations": _observations(1.10),
                }
                for workload_id, digest in WORKLOADS.items()
            ],
        },
    }


class QualificationContractTest(unittest.TestCase):
    def test_01_complete_receipt_passes(self):
        audit = audit_receipt(_receipt())
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["performance"]["aggregate_geomean"], 1.10)

    def test_02_provenance_drift_fails_closed(self):
        receipt = _receipt()
        receipt["provenance"]["flashinfer_commit"] = "b" * 40
        with self.assertRaisesRegex(QualificationError, "flashinfer_commit"):
            audit_receipt(receipt)

    def test_03_prompt_exactly_once_gate(self):
        receipt = _receipt()
        receipt["accuracy"]["arms"]["candidate"]["prompts"][37]["request_count"] = 2
        with self.assertRaisesRegex(QualificationError, "exactly once"):
            audit_receipt(receipt)

    def test_04_score_no_drop_and_kl_gates(self):
        with self.subTest("candidate score drop"):
            receipt = _receipt()
            receipt["accuracy"]["arms"]["baseline"] = _accuracy_arm(1224)
            with self.assertRaisesRegex(QualificationError, "accuracy dropped"):
                audit_receipt(receipt)
        with self.subTest("KL threshold"):
            receipt = _receipt()
            value = math.exp(0.1) - 1.0 - 0.1
            for row in receipt["accuracy"]["kl"]["records"]:
                row["baseline_logprobs"] = [0.0]
                row["candidate_logprobs"] = [-0.1]
            receipt["accuracy"]["kl"]["mean_kl"] = value
            with self.assertRaisesRegex(QualificationError, "not < 0.0035"):
                audit_receipt(receipt)

    def test_05_tp4_exact_route_and_zero_baseline_gate(self):
        receipt = _receipt()
        baseline_rank = receipt["routes"]["arms"]["baseline"][2]
        baseline_rank["decode_routes"] = ["cake.gdn_decode.noncp.tile16_fullwarp"]
        baseline_rank["cake_route_count"] = 1
        with self.assertRaisesRegex(QualificationError, "baseline rank 2 recorded"):
            audit_receipt(receipt)

    def test_06_abba_eight_observations_gate(self):
        receipt = _receipt()
        receipt["performance"]["workloads"][0]["observations"][1]["arm"] = "baseline"
        with self.assertRaisesRegex(QualificationError, "ABBA"):
            audit_receipt(receipt)

    def test_07_aggregate_geomean_and_lower_ci_gate(self):
        receipt = _receipt()
        for workload in receipt["performance"]["workloads"]:
            workload["observations"] = _observations(1.0)
        with self.assertRaisesRegex(QualificationError, "geomean .* is not > 1"):
            audit_receipt(receipt)

    def test_08_resolved_workload_regression_gate(self):
        receipt = _receipt()
        receipt["performance"]["workloads"][0]["observations"] = _observations(0.95)
        receipt["performance"]["workloads"][1]["observations"] = _observations(1.30)
        with self.assertRaisesRegex(QualificationError, "resolved regression"):
            audit_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
