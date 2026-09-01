from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from tools.gdn_public_qualification.collect import ROUTE_RE, collect_accuracy
from tools.gdn_public_qualification.accuracy_server_hook import (
    AUTHORITY_ENV as ACCURACY_AUTHORITY_ENV,
    AUTHORITY_SHA256_ENV as ACCURACY_AUTHORITY_SHA256_ENV,
    begin_request as begin_accuracy_server_request,
    finish_request as finish_accuracy_server_request,
    load_authority as load_accuracy_server_authority,
)
from tools.gdn_public_qualification.contract import (
    ACCURACY_DECODE_PARAMS,
    ACCURACY_SAMPLING_PARAMS,
    ACCURACY_SERVER_AUTHORITY_SCHEMA,
    ACCURACY_SERVER_RECEIPT_SCHEMA,
    ABBA_ORDER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXACT_T4_ROUTE,
    HASHES,
    KL_DIRECTION,
    KL_DISTRIBUTION_SCHEMA,
    KL_METRIC,
    KL_NORMALIZATION_ATOL,
    KL_POSITION_AGGREGATION,
    KL_SAMPLE_AGGREGATION,
    KL_TOKEN_ID_ORDER,
    KL_VOCAB_CHUNK_SIZE,
    MTP_PROBE_MAX_NEW_TOKENS,
    MTP_PROBE_PROMPT_INDEX,
    MTP_SPECULATIVE_EAGLE_TOPK,
    MTP_SPECULATIVE_NUM_DRAFT_TOKENS,
    MTP_SPECULATIVE_NUM_STEPS,
    MODEL_INFO_MANIFEST_FILE_COUNT_KEY,
    MODEL_INFO_MANIFEST_SHA256_KEY,
    MODEL_MANIFEST_FILE_COUNT,
    PROMPT_COUNT,
    PLAN_SCHEMA,
    SCHEMA,
    WORKLOADS,
    QualificationError,
    audit_receipt,
    canonical_json_sha256,
    canonical_json_text,
    expected_provenance,
    expected_server_config,
    load_strict_json,
    qualification_checkout_identity,
    verify_model_directory_against_manifest,
)
from tools.gdn_public_qualification.render_plan import (
    _server_command,
    _server_hosts,
    _write_plan_exclusive,
)
from tools.gdn_public_qualification.kl_sink_hook import marker_for_sample
from tools.gdn_public_qualification.kl_sink_server import (
    prepare_accuracy_request_root,
    prepare_sink_root,
)


TEST_CAMPAIGN_ID = ""
TEST_EVIDENCE_ROOT: Path | None = None
TEST_PLAN_SPEC: dict[str, str] | None = None


def _prompt_rows(arm: str, correct_count: int) -> list[dict]:
    rows = []
    for index in range(PROMPT_COUNT):
        answer_value = index if index < correct_count else index + 1
        output_ids = [answer_value]
        rows.append({
            "question_index": index,
            "source_row_index": index + 5,
            "request_count": 1,
            "request_id": f"gdn-gsm8k-{TEST_CAMPAIGN_ID}-{arm}-{index:04d}",
            "input_ids_sha256": hashlib.sha256(
                json.dumps([index + 1], separators=(",", ":")).encode()
            ).hexdigest(),
            "input_token_count": 1,
            "output_ids_sha256": hashlib.sha256(
                json.dumps(output_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "response": {
                "text": f"answer {answer_value}",
                "output_ids": output_ids,
                "meta_info": {
                    "id": f"gdn-gsm8k-{TEST_CAMPAIGN_ID}-{arm}-{index:04d}"
                },
            },
            "decoded_text": f"answer {answer_value}",
            "decode_params": dict(ACCURACY_DECODE_PARAMS),
            "correct": index < correct_count,
        })
    return rows


def _accuracy_arm(arm: str, correct_count: int) -> dict:
    if TEST_EVIDENCE_ROOT is None:
        raise RuntimeError("accuracy test evidence root is not initialized")
    prompt_rows = _prompt_rows(arm, correct_count)
    ledger_path = TEST_EVIDENCE_ROOT / f"gsm8k-{arm}-{correct_count}-request-ledger.jsonl"
    events = [
        {
            "schema": "gdn-gsm8k-request-ledger-v1",
            "arm": arm,
            "campaign_id": TEST_CAMPAIGN_ID,
            "prompt_count": PROMPT_COUNT,
        }
    ]
    for index, row in enumerate(prompt_rows):
        request_id = f"gdn-gsm8k-{TEST_CAMPAIGN_ID}-{arm}-{index:04d}"
        payload = {
            "rid": request_id,
            "input_ids": [index + 1],
            "sampling_params": ACCURACY_SAMPLING_PARAMS,
        }
        events.extend(
            (
                {
                    "event": "dispatch",
                    "question_index": index,
                    "request_id": request_id,
                    "payload_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                },
                {
                    "event": "response",
                    "question_index": index,
                    "request_id": request_id,
                    "response_sha256": hashlib.sha256(
                        json.dumps(row["response"], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                },
            )
        )
    ledger_bytes = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    ).encode()
    if ledger_path.exists():
        if ledger_path.read_bytes() != ledger_bytes:
            raise RuntimeError(f"test request ledger drift: {ledger_path}")
    else:
        ledger_path.write_bytes(ledger_bytes)
    server_root = TEST_EVIDENCE_ROOT / f"server-{arm}-{correct_count}" / "accuracy_requests"
    server_root.mkdir(parents=True, exist_ok=True)
    authority = {
        "schema": ACCURACY_SERVER_AUTHORITY_SCHEMA,
        "root": str(server_root),
        "arm": arm,
        "prompt_count": PROMPT_COUNT,
        "accepted_count_per_prompt": 1,
        "completed_count_per_prompt": 1,
    }
    authority_path = server_root / "authority.json"
    authority_bytes = (canonical_json_text(authority) + "\n").encode()
    if authority_path.exists() and authority_path.read_bytes() != authority_bytes:
        raise RuntimeError(f"test server authority drift: {authority_path}")
    authority_path.write_bytes(authority_bytes)
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    receipt = {
        "schema": ACCURACY_SERVER_RECEIPT_SCHEMA,
        "authority_sha256": authority_sha256,
        "arm": arm,
        "campaign_id": TEST_CAMPAIGN_ID,
        "prompt_count": PROMPT_COUNT,
        "records": [
            {
                "question_index": index,
                "request_id": f"gdn-gsm8k-{TEST_CAMPAIGN_ID}-{arm}-{index:04d}",
                "accepted_count": 1,
                "completed_count": 1,
                "response_sha256": hashlib.sha256(
                    json.dumps(
                        row["response"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            for index, row in enumerate(prompt_rows)
        ],
    }
    receipt_path = server_root / "receipt.json"
    receipt_bytes = (canonical_json_text(receipt) + "\n").encode()
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
        raise RuntimeError(f"test server receipt drift: {receipt_path}")
    receipt_path.write_bytes(receipt_bytes)
    return {
        "arm": arm,
        "campaign_id": TEST_CAMPAIGN_ID,
        "plan_sha256": TEST_CAMPAIGN_ID,
        "dataset_sha256": HASHES["gsm8k_dataset_sha256"],
        "prompt_ids_sha256": HASHES["gsm8k_prompt_ids_sha256"],
        "request_payload": "input_ids",
        "sampling_params": dict(ACCURACY_SAMPLING_PARAMS),
        "server_config": expected_server_config(arm),
        "model_identity": {
            "model_path": "/sealed/model",
            "tokenizer_path": "/sealed/model",
            "model_manifest_sha256": HASHES["model_manifest_sha256"],
            "model_manifest_file_count": MODEL_MANIFEST_FILE_COUNT,
        },
        "request_ledger": {
            "path": ledger_path.name,
            "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        },
        "server_request_evidence": {
            "authority": {
                "path": str(authority_path.relative_to(TEST_EVIDENCE_ROOT)),
                "sha256": authority_sha256,
            },
            "receipt": {
                "path": str(receipt_path.relative_to(TEST_EVIDENCE_ROOT)),
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            },
        },
        "score": correct_count / PROMPT_COUNT,
        "prompts": prompt_rows,
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


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_kl_evidence(
    root: Path,
    baseline_probabilities: tuple[float, float] = (0.5, 0.5),
    candidate_probabilities: tuple[float, float] = (0.5, 0.5),
) -> tuple[dict[str, dict[str, str]], float]:
    root.mkdir(parents=True)
    manifest_specs = {}
    baseline_manifest_sha = None
    float_rows = {}
    for arm, probabilities in (
        ("baseline", baseline_probabilities),
        ("candidate", candidate_probabilities),
    ):
        arm_root = root / arm
        shard_root = arm_root / "shards"
        shard_root.mkdir(parents=True)
        logprobs = tuple(math.log(value) for value in probabilities)
        float_rows[arm] = struct.unpack("<2f", struct.pack("<2f", *logprobs))
        records = []
        for sample_index in range(48):
            shard_path = shard_root / f"sample-{sample_index:03d}.f32le"
            shard_path.write_bytes(struct.pack("<2f", *logprobs))
            output_ids = [1]
            records.append(
                {
                    "sample_index": sample_index,
                    "input_ids_sha256": hashlib.sha256(f"input-{sample_index}".encode()).hexdigest(),
                    "output_ids": output_ids,
                    "output_ids_sha256": hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest(),
                    "position_count": 1,
                    "shards": [
                        {
                            "path": f"shards/{shard_path.name}",
                            "token_start": 0,
                            "token_end": 2,
                            "shape": [1, 2],
                            "byte_count": 8,
                            "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        manifest = {
            "schema": KL_DISTRIBUTION_SCHEMA,
            "arm": arm,
            "sample_count": 48,
            "input_ids_sha256": HASHES["longbench_first48_ids_sha256"],
            "model_manifest_sha256": HASHES["model_manifest_sha256"],
            "model_path": "/sealed/model",
            "tokenizer_path": "/sealed/model",
            "vocab_size": 2,
            "token_id_order": KL_TOKEN_ID_ORDER,
            "dtype": "float32",
            "byte_order": "little",
            "normalization_atol": KL_NORMALIZATION_ATOL,
            "vocab_chunk_size": KL_VOCAB_CHUNK_SIZE,
            "sink_authority_sha256": ("a" if arm == "baseline" else "b") * 64,
            "records": records,
        }
        if arm == "candidate":
            manifest["reference_manifest_sha256"] = baseline_manifest_sha
        manifest_path = arm_root / "manifest.json"
        digest = _write_json(manifest_path, manifest)
        if arm == "baseline":
            baseline_manifest_sha = digest
        manifest_specs[f"{arm}_manifest"] = {
            "path": str(manifest_path.relative_to(root)),
            "sha256": digest,
        }
    baseline_values = float_rows["baseline"]
    candidate_values = float_rows["candidate"]
    kl_value = sum(
        math.exp(p_value) * (p_value - q_value)
        for p_value, q_value in zip(baseline_values, candidate_values)
    )
    return manifest_specs, max(0.0, kl_value)


def _receipt(kl_specs: dict[str, dict[str, str]], kl_value: float = 0.0) -> dict:
    if TEST_PLAN_SPEC is None:
        raise RuntimeError("accuracy test plan is not initialized")
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
            "campaign_id": TEST_CAMPAIGN_ID,
            "plan": dict(TEST_PLAN_SPEC),
            "dataset": {
                "path": "gsm8k.jsonl",
                "sha256": HASHES["gsm8k_dataset_sha256"],
            },
            "prompt_ids": {
                "path": "gsm8k-prompt-token-ids.json",
                "sha256": HASHES["gsm8k_prompt_ids_sha256"],
            },
            "arms": {
                "baseline": _accuracy_arm("baseline", 1223),
                "candidate": _accuracy_arm("candidate", 1223),
            },
            "kl": {
                "metric": KL_METRIC,
                "direction": KL_DIRECTION,
                "position_aggregation": KL_POSITION_AGGREGATION,
                "sample_aggregation": KL_SAMPLE_AGGREGATION,
                "mean_sample_kl": kl_value,
                "max_sample_kl": kl_value,
                "max_position_kl": kl_value,
                **kl_specs,
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
    def setUp(self):
        global TEST_CAMPAIGN_ID, TEST_EVIDENCE_ROOT, TEST_PLAN_SPEC
        self.temp_directory = tempfile.TemporaryDirectory()
        self.evidence_root = Path(self.temp_directory.name)
        self.kl_specs, self.kl_value = _write_kl_evidence(self.evidence_root / "default")
        self.original_dataset_sha256 = HASHES["gsm8k_dataset_sha256"]
        self.original_prompt_ids_sha256 = HASHES["gsm8k_prompt_ids_sha256"]
        dataset_path = self.evidence_root / "default" / "gsm8k.jsonl"
        dataset_path.write_text(
            "".join(
                json.dumps(
                    {
                        "question": f"question {source_index}",
                        "answer": f"#### {max(0, source_index - 5)}",
                    },
                    sort_keys=True,
                )
                + "\n"
                for source_index in range(5 + PROMPT_COUNT)
            )
        )
        HASHES["gsm8k_dataset_sha256"] = hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest()
        prompt_ids = [[index + 1] for index in range(PROMPT_COUNT)]
        HASHES["gsm8k_prompt_ids_sha256"] = _write_json(
            self.evidence_root / "default" / "gsm8k-prompt-token-ids.json",
            prompt_ids,
        )
        TEST_EVIDENCE_ROOT = self.evidence_root / "default"
        plan_provenance = expected_provenance()
        plan_provenance.update(
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
        plan = {
            "schema": PLAN_SCHEMA,
            "provenance": plan_provenance,
            "bindings": {
                "model_path": "/sealed/model",
                "tokenizer_path": "/sealed/model",
                "artifacts": {"model_manifest": "/sealed/model-manifest.json"},
            },
            "accuracy": {
                "prompt_count": PROMPT_COUNT,
                "shots": 5,
                "requests_per_prompt_per_arm": 1,
                "minimum_score": 0.93,
                "candidate_no_drop": True,
                "server_request_authority_schema": ACCURACY_SERVER_AUTHORITY_SCHEMA,
                "server_request_receipt_schema": ACCURACY_SERVER_RECEIPT_SCHEMA,
                "server_request_hook": "tools.gdn_public_qualification.accuracy_server_hook",
                "server_duplicate_request_policy": "reject_before_generation",
                "server_model_manifest_observation": {
                    "sha256_key": MODEL_INFO_MANIFEST_SHA256_KEY,
                    "file_count_key": MODEL_INFO_MANIFEST_FILE_COUNT_KEY,
                },
            },
        }
        plan_path = TEST_EVIDENCE_ROOT / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        TEST_CAMPAIGN_ID = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        TEST_PLAN_SPEC = {"path": plan_path.name, "sha256": TEST_CAMPAIGN_ID}
        self.decode_patch = patch(
            "tools.gdn_public_qualification.contract.decode_accuracy_output_ids",
            side_effect=lambda tokenizer_path, output_ids: f"answer {output_ids[-1]}",
        )
        self.decode_patch.start()
        self.checkout_patch = patch(
            "tools.gdn_public_qualification.contract.qualification_checkout_identity",
            return_value=("a" * 40, "b" * 40),
        )
        self.checkout_patch.start()

    def tearDown(self):
        global TEST_CAMPAIGN_ID, TEST_EVIDENCE_ROOT, TEST_PLAN_SPEC
        HASHES["gsm8k_dataset_sha256"] = self.original_dataset_sha256
        HASHES["gsm8k_prompt_ids_sha256"] = self.original_prompt_ids_sha256
        TEST_CAMPAIGN_ID = ""
        TEST_EVIDENCE_ROOT = None
        TEST_PLAN_SPEC = None
        self.checkout_patch.stop()
        self.decode_patch.stop()
        self.temp_directory.cleanup()

    def _audit(self, receipt: dict, root: Path | None = None) -> dict:
        default_root = self.evidence_root / "default"
        selected_root = root or default_root
        if selected_root != default_root:
            selected_root.mkdir(parents=True, exist_ok=True)
            shared_specs = [
                receipt["accuracy"]["plan"],
                receipt["accuracy"]["dataset"],
                receipt["accuracy"]["prompt_ids"],
                receipt["accuracy"]["arms"]["baseline"]["request_ledger"],
                receipt["accuracy"]["arms"]["candidate"]["request_ledger"],
                receipt["accuracy"]["arms"]["baseline"]["server_request_evidence"]["authority"],
                receipt["accuracy"]["arms"]["baseline"]["server_request_evidence"]["receipt"],
                receipt["accuracy"]["arms"]["candidate"]["server_request_evidence"]["authority"],
                receipt["accuracy"]["arms"]["candidate"]["server_request_evidence"]["receipt"],
            ]
            for spec in shared_specs:
                source = default_root / spec["path"]
                destination = selected_root / spec["path"]
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
        return audit_receipt(receipt, selected_root)

    def _rewrite_manifest(self, arm: str, mutate) -> None:
        root = self.evidence_root / "default"
        spec = self.kl_specs[f"{arm}_manifest"]
        path = root / spec["path"]
        manifest = json.loads(path.read_text())
        mutate(manifest)
        spec["sha256"] = _write_json(path, manifest)
        if arm == "baseline":
            candidate_spec = self.kl_specs["candidate_manifest"]
            candidate_path = root / candidate_spec["path"]
            candidate = json.loads(candidate_path.read_text())
            candidate["reference_manifest_sha256"] = spec["sha256"]
            candidate_spec["sha256"] = _write_json(candidate_path, candidate)

    def test_01_complete_receipt_passes(self):
        audit = self._audit(_receipt(self.kl_specs, self.kl_value))
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["performance"]["aggregate_geomean"], 1.10)

    def test_02_provenance_drift_fails_closed(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["provenance"]["flashinfer_commit"] = "b" * 40
        with self.assertRaisesRegex(QualificationError, "flashinfer_commit"):
            self._audit(receipt)
        with self.subTest("plan and receipt qualification source"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["provenance"]["qualification_commit"] = "c" * 40
            with self.assertRaisesRegex(
                QualificationError, "plan provenance differs from receipt provenance"
            ):
                self._audit(receipt)
        with self.subTest("qualification source must match actual auditor checkout"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            with patch(
                "tools.gdn_public_qualification.contract.qualification_checkout_identity",
                return_value=("c" * 40, "b" * 40),
            ):
                with self.assertRaisesRegex(
                    QualificationError,
                    "qualification commit differs from the auditor checkout",
                ):
                    self._audit(receipt)

    def test_02b_checkout_identity_uses_real_clean_git_objects(self):
        checkout = self.evidence_root / "exact-checkout"
        checkout.mkdir()
        (checkout / "tracked.txt").write_text("sealed\n")
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Qualification Test",
                "-c",
                "user.email=qualification@example.invalid",
                "commit",
                "-q",
                "-m",
                "sealed checkout",
            ],
            cwd=checkout,
            check=True,
        )
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        expected_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=checkout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertEqual(
            qualification_checkout_identity(checkout),
            (expected_commit, expected_tree),
        )
        (checkout / "untracked.txt").write_text("dirty\n")
        with self.assertRaisesRegex(QualificationError, "completely clean"):
            qualification_checkout_identity(checkout)

    def test_03_prompt_exactly_once_gate(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["accuracy"]["arms"]["candidate"]["prompts"][37]["request_count"] = 2
        with self.assertRaisesRegex(QualificationError, "exactly once"):
            self._audit(receipt)
        with self.subTest("sealed input IDs"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["prompts"][37]["input_ids_sha256"] = "0" * 64
            with self.assertRaisesRegex(QualificationError, "input IDs differ"):
                self._audit(receipt)
        with self.subTest("token-ID payload"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["request_payload"] = "text"
            with self.assertRaisesRegex(QualificationError, "did not send sealed token IDs"):
                self._audit(receipt)
        with self.subTest("dataset authority"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["dataset_sha256"] = "0" * 64
            with self.assertRaisesRegex(QualificationError, "dataset authority differs"):
                self._audit(receipt)
        with self.subTest("rendered plan authority"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["plan"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(QualificationError, "campaign plan spec differs"):
                self._audit(receipt)
        with self.subTest("request ledger authority"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["request_ledger"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(QualificationError, "request ledger SHA256 mismatch"):
                self._audit(receipt)
        with self.subTest("server exact-once authority"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            evidence = receipt["accuracy"]["arms"]["candidate"][
                "server_request_evidence"
            ]
            source = self.evidence_root / "default" / evidence["receipt"]["path"]
            value = json.loads(source.read_text())
            value["records"][37]["accepted_count"] = 2
            changed = self.evidence_root / "default" / "candidate-invalid-server-receipt.json"
            changed.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            evidence["receipt"] = {
                "path": changed.name,
                "sha256": hashlib.sha256(changed.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(QualificationError, "exact-once acceptance"):
                self._audit(receipt)
        with self.subTest("arm identity"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["arm"] = "baseline"
            with self.assertRaisesRegex(QualificationError, "arm identity differs"):
                self._audit(receipt)
        with self.subTest("request identity"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["prompts"][37][
                "request_id"
            ] = f"gdn-gsm8k-{TEST_CAMPAIGN_ID}-candidate-0036"
            with self.assertRaisesRegex(QualificationError, "request ID differs"):
                self._audit(receipt)
        with self.subTest("server backend identity"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["server_config"] = (
                expected_server_config("baseline")
            )
            with self.assertRaisesRegex(QualificationError, "server configuration differs"):
                self._audit(receipt)
        with self.subTest("model identity must match rendered plan"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["model_identity"][
                "model_path"
            ] = "/different/model"
            with self.assertRaisesRegex(QualificationError, "differs from the campaign plan"):
                self._audit(receipt)
        with self.subTest("sampling parameters"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["sampling_params"][
                "temperature"
            ] = 0.5
            with self.assertRaisesRegex(QualificationError, "sampling parameters differ"):
                self._audit(receipt)
        with self.subTest("raw-response rescoring"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["prompts"][37]["response"][
                "text"
            ] = "answer 999999"
            with self.assertRaisesRegex(QualificationError, "ledger response differs"):
                self._audit(receipt)
        with self.subTest("source-owned decoding"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            prompt = receipt["accuracy"]["arms"]["candidate"]["prompts"][37]
            prompt["decoded_text"] = "answer 999999"
            with self.assertRaisesRegex(QualificationError, "source-owned output decoding differs"):
                self._audit(receipt)
        with self.subTest("strict boolean"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["candidate"]["prompts"][37]["correct"] = 1
            with self.assertRaisesRegex(QualificationError, "invalid correctness"):
                self._audit(receipt)
        with self.subTest("ledger response must follow dispatch"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            spec = receipt["accuracy"]["arms"]["candidate"]["request_ledger"]
            source = self.evidence_root / "default" / spec["path"]
            lines = source.read_text().splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            reordered = self.evidence_root / "default" / "candidate-reordered-ledger.jsonl"
            reordered.write_text("\n".join(lines) + "\n")
            spec["path"] = reordered.name
            spec["sha256"] = hashlib.sha256(reordered.read_bytes()).hexdigest()
            with self.assertRaisesRegex(QualificationError, "response precedes dispatch"):
                self._audit(receipt)

    def test_03b_canonical_json_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            canonical_json_text({"value": float("nan")})
        with self.assertRaises(ValueError):
            load_strict_json('{"value":NaN}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            load_strict_json('{"outer":{"value":1,"value":2}}')

    def test_03c_rendered_plan_output_is_immutable(self):
        path = self.evidence_root / "immutable-plan.json"
        _write_plan_exclusive(path, {"schema": PLAN_SCHEMA})
        with self.assertRaises(FileExistsError):
            _write_plan_exclusive(path, {"schema": "replacement"})

    def test_04_score_no_drop_and_kl_gates(self):
        with self.subTest("candidate score drop"):
            receipt = _receipt(self.kl_specs, self.kl_value)
            receipt["accuracy"]["arms"]["baseline"] = _accuracy_arm("baseline", 1224)
            with self.assertRaisesRegex(QualificationError, "accuracy dropped"):
                self._audit(receipt)
        with self.subTest("KL threshold"):
            threshold_root = self.evidence_root / "threshold"
            specs, value = _write_kl_evidence(
                threshold_root,
                baseline_probabilities=(0.5, 0.5),
                candidate_probabilities=(0.55, 0.45),
            )
            receipt = _receipt(specs, value)
            with self.assertRaisesRegex(QualificationError, "maximum sample full-vocabulary KL .*not < 0.0035"):
                self._audit(receipt, threshold_root)

    def test_05_tp4_exact_route_and_zero_baseline_gate(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
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
            self._audit(receipt)

    def test_06_candidate_must_prove_exact_t4_route(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["routes"]["arms"]["candidate"][1]["route_observations"] = [
            {
                "route": receipt["routes"]["expected_candidate_routes"]["prefill"][0],
                "phase": "prefill",
                "t": 128,
                "gates_present": True,
            }
        ]
        with self.assertRaisesRegex(QualificationError, "exact optimized T=4 route"):
            self._audit(receipt)

    def test_07_mtp_probe_requires_exact_resolved_server_config(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["mtp_probe"]["arms"]["candidate"]["server_config"][
            "linear_attn_verify_backend"
        ] = "triton"
        with self.assertRaisesRegex(QualificationError, "server configuration differs"):
            self._audit(receipt)

    def test_08_abba_eight_observations_gate(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["performance"]["workloads"][0]["observations"][1]["arm"] = "baseline"
        with self.assertRaisesRegex(QualificationError, "ABBA"):
            self._audit(receipt)

    def test_09_aggregate_geomean_and_lower_ci_gate(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        for workload in receipt["performance"]["workloads"]:
            workload["observations"] = _observations(1.0)
        with self.assertRaisesRegex(QualificationError, "geomean .* is not > 1"):
            self._audit(receipt)

    def test_10_resolved_workload_regression_gate(self):
        receipt = _receipt(self.kl_specs, self.kl_value)
        receipt["performance"]["workloads"][0]["observations"] = _observations(0.95)
        receipt["performance"]["workloads"][1]["observations"] = _observations(1.30)
        with self.assertRaisesRegex(QualificationError, "resolved regression"):
            self._audit(receipt)

    def test_11_server_hosts_are_per_arm_and_fail_closed(self):
        binding = {
            "server_hosts": {
                "baseline": "gb300-a.internal",
                "candidate": "gb300-b.internal",
            },
            "ports": {"baseline": 31000, "candidate": 31001},
            "model_path": "/model",
            "tokenizer_path": "/model",
            "vocab_size": 151936,
            "kl_sink_roots": {
                "baseline": "/sink/baseline",
                "candidate": "/sink/candidate",
            },
            "artifacts": {"model_manifest": "/sealed/model-files.json"},
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
            self.assertEqual(
                command[command.index("--model-manifest") + 1],
                "/sealed/model-files.json",
            )
            self.assertEqual(
                command[command.index("--verified-model-path") + 1], "/model"
            )
            self.assertEqual(
                command[command.index("--verified-tokenizer-path") + 1],
                "/model",
            )
            self.assertIn("tools.gdn_public_qualification.kl_sink_server", command)

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

    def test_13_full_vocabulary_coverage_fails_closed(self):
        self._rewrite_manifest(
            "baseline",
            lambda manifest: manifest["records"][0]["shards"][0].update(
                {"token_end": 1, "shape": [1, 1], "byte_count": 4}
            ),
        )
        with self.assertRaisesRegex(QualificationError, "does not cover the full vocabulary"):
            self._audit(_receipt(self.kl_specs, self.kl_value))

    def test_14_tokenizer_and_position_alignment_fail_closed(self):
        with self.subTest("tokenizer"):
            self._rewrite_manifest(
                "candidate",
                lambda manifest: manifest.update({"tokenizer_path": "/different/tokenizer"}),
            )
            with self.assertRaisesRegex(QualificationError, "tokenizer_path alignment differs"):
                self._audit(_receipt(self.kl_specs, self.kl_value))

        # setUp is not re-entered between subtests, so restore fresh evidence in
        # a separate root for the independent position-alignment case.
        position_root = self.evidence_root / "position"
        specs, value = _write_kl_evidence(position_root)
        candidate_path = position_root / specs["candidate_manifest"]["path"]
        candidate = json.loads(candidate_path.read_text())
        candidate["records"][0]["output_ids"] = [0]
        candidate["records"][0]["output_ids_sha256"] = hashlib.sha256(b"[0]").hexdigest()
        specs["candidate_manifest"]["sha256"] = _write_json(candidate_path, candidate)
        with self.assertRaisesRegex(QualificationError, "output_ids_sha256 alignment differs"):
            self._audit(_receipt(specs, value), position_root)

    def test_15_probability_mass_must_be_normalized(self):
        mass_root = self.evidence_root / "mass"
        specs, value = _write_kl_evidence(
            mass_root,
            baseline_probabilities=(0.4, 0.4),
            candidate_probabilities=(0.4, 0.4),
        )
        with self.assertRaisesRegex(QualificationError, "baseline probability mass .* is not normalized"):
            self._audit(_receipt(specs, value), mass_root)

    def test_16_kl_direction_and_aggregation_are_exact(self):
        for key, invalid in (
            ("direction", "Q_candidate||P_baseline"),
            ("position_aggregation", "maximum_over_positions"),
            ("sample_aggregation", "mean_across_samples"),
        ):
            with self.subTest(key=key):
                receipt = _receipt(self.kl_specs, self.kl_value)
                receipt["accuracy"]["kl"][key] = invalid
                with self.assertRaisesRegex(QualificationError, key.replace("_", " ")):
                    self._audit(receipt)

    def test_17_distribution_shard_hash_drift_fails_closed(self):
        shard = self.evidence_root / "default" / "baseline" / "shards" / "sample-000.f32le"
        shard.write_bytes(struct.pack("<2f", math.log(0.6), math.log(0.4)))
        with self.assertRaisesRegex(QualificationError, "shard 0 SHA256 mismatch"):
            self._audit(_receipt(self.kl_specs, self.kl_value))

    def test_18_sink_root_is_fresh_sealed_and_request_marker_has_no_path(self):
        root = self.evidence_root / "sink-server"
        authority_path, authority_sha = prepare_sink_root(root, "baseline", 151936)
        self.assertTrue(authority_path.is_file())
        self.assertEqual(hashlib.sha256(authority_path.read_bytes()).hexdigest(), authority_sha)
        marker = marker_for_sample(151936, 17)
        self.assertEqual(marker, [151935, 151934, 151935, 151934, 17])
        self.assertTrue(all(type(value) is int for value in marker))
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_sink_root(root, "baseline", 151936)

    def test_18a_server_hashes_exact_list_shaped_model_manifest(self):
        model_root = self.evidence_root / "sealed-model"
        model_root.mkdir()
        rows = []
        for index in range(MODEL_MANIFEST_FILE_COUNT):
            name = f"model-file-{index:02d}.bin"
            payload = f"sealed payload {index}\n".encode()
            (model_root / name).write_bytes(payload)
            rows.append(
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest = self.evidence_root / "model-files.json"
        manifest.write_text(json.dumps(rows, sort_keys=True) + "\n")
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        with patch.dict(
            HASHES, {"model_manifest_sha256": manifest_sha256}, clear=False
        ):
            self.assertEqual(
                verify_model_directory_against_manifest(
                    model_root, model_root, manifest
                ),
                {
                    "manifest_sha256": manifest_sha256,
                    "file_count": MODEL_MANIFEST_FILE_COUNT,
                },
            )
            (model_root / rows[0]["path"]).write_bytes(b"tampered\n")
            with self.assertRaisesRegex(QualificationError, "byte count differs"):
                verify_model_directory_against_manifest(
                    model_root, model_root, manifest
                )

    def test_18aa_server_rejects_unsealed_model_manifest_shapes(self):
        model_root = self.evidence_root / "manifest-shape-model"
        model_root.mkdir()
        rows = []
        for index in range(MODEL_MANIFEST_FILE_COUNT):
            name = f"file-{index:02d}"
            payload = bytes([index])
            (model_root / name).write_bytes(payload)
            rows.append(
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest = self.evidence_root / "invalid-model-files.json"
        manifest.write_text(json.dumps({"files": rows}, sort_keys=True) + "\n")
        with patch.dict(
            HASHES,
            {"model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
            clear=False,
        ):
            with self.assertRaisesRegex(QualificationError, "top-level 18-row list"):
                verify_model_directory_against_manifest(
                    model_root, model_root, manifest
                )

        manifest.write_text(json.dumps(rows, sort_keys=True) + "\n")
        extra = model_root / "not-in-manifest"
        extra.write_text("extra\n")
        with patch.dict(
            HASHES,
            {"model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
            clear=False,
        ):
            with self.assertRaisesRegex(QualificationError, "file set differs"):
                verify_model_directory_against_manifest(
                    model_root, model_root, manifest
                )
            with self.assertRaisesRegex(QualificationError, "same sealed directory"):
                verify_model_directory_against_manifest(
                    model_root, self.evidence_root, manifest
                )

    def test_18b_server_hook_atomically_rejects_concurrent_duplicate(self):
        root = self.evidence_root / "server-exact-once"
        prepare_sink_root(root, "candidate", 151936)
        authority_path, authority_sha = prepare_accuracy_request_root(root, "candidate")
        request = SimpleNamespace(
            rid=f"gdn-gsm8k-{'d' * 64}-candidate-0000",
            stream=False,
        )
        load_accuracy_server_authority.cache_clear()
        with patch.dict(
            "os.environ",
            {
                ACCURACY_AUTHORITY_ENV: str(authority_path),
                ACCURACY_AUTHORITY_SHA256_ENV: authority_sha,
            },
            clear=False,
        ):
            barrier = Barrier(2)

            def admit():
                barrier.wait()
                try:
                    return begin_accuracy_server_request(request)
                except ValueError as exc:
                    return exc

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(lambda _: admit(), range(2)))
            self.assertEqual(sum(isinstance(value, dict) for value in outcomes), 1)
            failures = [value for value in outcomes if isinstance(value, ValueError)]
            self.assertEqual(len(failures), 1)
            self.assertIn("duplicate qualification", str(failures[0]))
        load_accuracy_server_authority.cache_clear()

    def test_18c_server_hook_publishes_complete_response_bound_receipt(self):
        root = self.evidence_root / "server-complete-receipt"
        prepare_sink_root(root, "baseline", 151936)
        authority_path, authority_sha = prepare_accuracy_request_root(root, "baseline")
        campaign_id = "e" * 64
        responses = []
        load_accuracy_server_authority.cache_clear()
        with patch.dict(
            "os.environ",
            {
                ACCURACY_AUTHORITY_ENV: str(authority_path),
                ACCURACY_AUTHORITY_SHA256_ENV: authority_sha,
            },
            clear=False,
        ):
            for index in range(PROMPT_COUNT):
                request_id = (
                    f"gdn-gsm8k-{campaign_id}-baseline-{index:04d}"
                )
                token = begin_accuracy_server_request(
                    SimpleNamespace(rid=request_id, stream=False)
                )
                response = {
                    "text": f"answer {index}",
                    "output_ids": [index],
                    "meta_info": {"id": request_id},
                }
                responses.append(response)
                finish_accuracy_server_request(token, response)
        load_accuracy_server_authority.cache_clear()
        receipt = load_strict_json(
            (root / "accuracy_requests" / "receipt.json").read_text()
        )
        self.assertEqual(receipt["authority_sha256"], authority_sha)
        self.assertEqual(receipt["campaign_id"], campaign_id)
        self.assertEqual(len(receipt["records"]), PROMPT_COUNT)
        for index, (record, response) in enumerate(
            zip(receipt["records"], responses)
        ):
            self.assertEqual(record["question_index"], index)
            self.assertEqual(record["accepted_count"], 1)
            self.assertEqual(record["completed_count"], 1)
            self.assertEqual(
                record["response_sha256"], canonical_json_sha256(response)
            )

    def test_19_accuracy_collector_rejects_caller_selected_dataset(self):
        with self.assertRaisesRegex(ValueError, "dataset hash differs"):
            collect_accuracy(SimpleNamespace(dataset_sha256="0" * 64))


if __name__ == "__main__":
    unittest.main()
