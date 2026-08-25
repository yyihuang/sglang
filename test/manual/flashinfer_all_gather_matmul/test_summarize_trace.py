#!/usr/bin/env python3
"""Static fixtures for exact four-rank CUPTI symbol gating."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import summarize_trace


SYMBOL = "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4"


class TraceGateTest(unittest.TestCase):
    def write_traces(self, root, names=None):
        names = names or [SYMBOL] * 4
        for rank, name in enumerate(names):
            path = root / f"agmm-candidate-1-TP-{rank}.trace.json.gz"
            payload = {
                "traceEvents": [
                    {"cat": "kernel", "name": name, "dur": 1.0},
                ]
            }
            with gzip.open(path, "wt") as stream:
                json.dump(payload, stream)

    def invoke(self, root):
        output = root / "evidence.json"
        argv = [
            "summarize_trace.py",
            "--trace-dir",
            str(root),
            "--expected-kernel-symbol",
            SYMBOL,
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            summarize_trace.main()
        return json.loads(output.read_text())

    def test_exact_symbol_on_all_four_ranks_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_traces(root)
            self.assertEqual(self.invoke(root)["rank_count"], 4)

    def test_prefixed_near_match_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_traces(root, ["prefix_" + SYMBOL, SYMBOL, SYMBOL, SYMBOL])
            with self.assertRaises(RuntimeError):
                self.invoke(root)


if __name__ == "__main__":
    unittest.main()
