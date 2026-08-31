from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest

from tools.gdn_public_qualification.collect import ROUTE_RE
from tools.gdn_public_qualification.contract import (
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXACT_T4_ROUTE,
    HASHES,
    KL_METRIC,
    MTP_PROBE_MAX_NEW_TOKENS,
    MTP_PROBE_PROMPT_INDEX,
    MTP_SPECULATIVE_EAGLE_TOPK,
    MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
    MTP_SPECULATIVE_NUM_STEPS,
    PROMPT_COUNT,
    SCHEMA,
    WORKLOADS,
    QualificationError,
    audit_receipt,
    expected_provenance,
)
from tools.gdn_public_qualification.render_plan import _server_command, _server_hosts


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


def _mtp_probe_arm(arm: str) -> dict:
    backend = "triton" if arm == "baseline" else "flashinfer"
    output_ids = list(range(10, 10 + MTP_PROBE_MAX_NEW_TOKENS))
    output_ids_sha256 = hashlib.sha256(
        json.dumps(output_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "arm": arm,
        "input_ids_sha256": HASHES["longbench_first48_ids_sha256"],
        "prompt_index": MTP_PROBE_PROMPT_INDEX,
        "request_count": 1,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": MTP_PROBE_MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "server_config": {
            "tp_size": 4,
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": MTP_SPECULATIVE_NUM_STEPS,
            "speculative_eagle_topk": MTP_SPECULATIVE_EAGLE_TOPK,
            "speculative_num_draft_tokens": MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
            "linear_attn_prefill_backend": backend,
            "linear_attn_decode_backend": backend,
            "linear_attn_verify_backend": backend,
        },
        "output_ids": output_ids,
        "output_ids_sha256": output_ids_sha256,
        "measured_runtime_seconds": 1.0,
    }


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
        "prefill": ["flashinfer.gdn_prefill.noncp.full_dv"],
        "decode": [EXACT_T4_ROUTE],
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
                "metric": KL_METRIC,
                "mean_kl": 0.0,
                "max_kl": 0.0,
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
        "mtp_probe": {
            "arms": {
                arm: _mtp_probe_arm(arm) for arm in ("baseline", "candidate")
            }
        },
        "routes": {
            "expected_candidate_routes": expected_routes,
            "arms": {
                "baseline": [
                    {
                        "rank": rank,
                        "prefill_routes": [],
                        "decode_routes": [],
                        "route_observations": [],
                        "marker_count": 0,
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
                        "route_observations": [
                            {
                                "route": expected_routes["prefill"][0],
                                "phase": "prefill",
                                "t": 128,
                                "gates_present": True,
                            },
                            {
                                "route": EXACT_T4_ROUTE,
                                "phase": "decode",
                                "t": 4,
                                "gates_present": True,
                            },
                        ],
                        "marker_count": 2,
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
            row = receipt["accuracy"]["kl"]["records"][0]
            row["baseline_logprobs"] = [0.0]
            row["candidate_logprobs"] = [-0.1]
            receipt["accuracy"]["kl"]["mean_kl"] = value / 48
            receipt["accuracy"]["kl"]["max_kl"] = value
            with self.assertRaisesRegex(QualificationError, "maximum sample .*not < 0.0035"):
                audit_receipt(receipt)

    def test_05_tp4_exact_route_and_zero_baseline_gate(self):
        receipt = _receipt()
        baseline_rank = receipt["routes"]["arms"]["baseline"][2]
        baseline_rank["decode_routes"] = [EXACT_T4_ROUTE]
        baseline_rank["route_observations"] = [
            {
                "route": EXACT_T4_ROUTE,
                "phase": "decode",
                "t": 4,
                "gates_present": True,
            }
        ]
        baseline_rank["marker_count"] = 1
        with self.assertRaisesRegex(QualificationError, "baseline rank 2 recorded optimized"):
            audit_receipt(receipt)

    def test_06_candidate_must_prove_exact_t4_route(self):
        receipt = _receipt()
        receipt["routes"]["arms"]["candidate"][1]["route_observations"] = [
            {
                "route": receipt["routes"]["expected_candidate_routes"]["prefill"][0],
                "phase": "prefill",
                "t": 128,
                "gates_present": True,
            }
        ]
        with self.assertRaisesRegex(QualificationError, "exact optimized T=4 route"):
            audit_receipt(receipt)

    def test_07_mtp_probe_requires_exact_resolved_server_config(self):
        receipt = _receipt()
        receipt["mtp_probe"]["arms"]["candidate"]["server_config"][
            "linear_attn_verify_backend"
        ] = "triton"
        with self.assertRaisesRegex(QualificationError, "server configuration differs"):
            audit_receipt(receipt)

    def test_08_abba_eight_observations_gate(self):
        receipt = _receipt()
        receipt["performance"]["workloads"][0]["observations"][1]["arm"] = "baseline"
        with self.assertRaisesRegex(QualificationError, "ABBA"):
            audit_receipt(receipt)

    def test_09_aggregate_geomean_and_lower_ci_gate(self):
        receipt = _receipt()
        for workload in receipt["performance"]["workloads"]:
            workload["observations"] = _observations(1.0)
        with self.assertRaisesRegex(QualificationError, "geomean .* is not > 1"):
            audit_receipt(receipt)

    def test_10_resolved_workload_regression_gate(self):
        receipt = _receipt()
        receipt["performance"]["workloads"][0]["observations"] = _observations(0.95)
        receipt["performance"]["workloads"][1]["observations"] = _observations(1.30)
        with self.assertRaisesRegex(QualificationError, "resolved regression"):
            audit_receipt(receipt)

    def test_11_server_hosts_are_per_arm_and_fail_closed(self):
        binding = {
            "server_hosts": {
                "baseline": "gb300-a.internal",
                "candidate": "gb300-b.internal",
            },
            "ports": {"baseline": 31000, "candidate": 31001},
            "model_path": "/model",
        }
        self.assertEqual(
            _server_hosts(binding),
            {"baseline": "gb300-a.internal", "candidate": "gb300-b.internal"},
        )
        baseline = _server_command(binding, "baseline")
        candidate = _server_command(binding, "candidate")
        self.assertEqual(baseline[baseline.index("--host") + 1], "gb300-a.internal")
        self.assertEqual(candidate[candidate.index("--host") + 1], "gb300-b.internal")
        for command, backend in ((baseline, "triton"), (candidate, "flashinfer")):
            self.assertEqual(
                command[command.index("--linear-attn-verify-backend") + 1], backend
            )
            self.assertEqual(
                command[command.index("--speculative-algorithm") + 1], "NEXTN"
            )
            self.assertEqual(
                command[command.index("--speculative-num-steps") + 1], "3"
            )
            self.assertEqual(
                command[command.index("--speculative-eagle-topk") + 1], "1"
            )
            self.assertEqual(
                command[command.index("--speculative-num-draft-tokens") + 1], "4"
            )

        invalid_hosts = (
            None,
            {"baseline": "gb300-a.internal"},
            {
                "baseline": "gb300-a.internal",
                "candidate": "gb300-b.internal",
                "extra": "gb300-c.internal",
            },
            {"baseline": "", "candidate": "gb300-b.internal"},
            {"baseline": "gb300-a.internal", "candidate": 7},
        )
        for hosts in invalid_hosts:
            with self.subTest(hosts=hosts), self.assertRaises(ValueError):
                _server_hosts({"server_hosts": hosts})

    def test_12_route_marker_parser_requires_neutral_attributable_schema(self):
        valid = (
            "INFO FLASHINFER_GDN_NONCP_ROUTE backend=gdn_noncp "
            f"route={EXACT_T4_ROUTE} phase=decode t=4 gates_present=True "
            "batch_size=16"
        )
        match = ROUTE_RE.search(valid)
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), (EXACT_T4_ROUTE, "decode", "decode", "4"))
        self.assertIsNone(
            ROUTE_RE.search(
                "FLASHINFER_GDN_NONCP_ROUTE backend=gdn_noncp "
                "route=cake.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp "
                "phase=decode t=4 gates_present=True"
            )
        )


if __name__ == "__main__":
    unittest.main()
