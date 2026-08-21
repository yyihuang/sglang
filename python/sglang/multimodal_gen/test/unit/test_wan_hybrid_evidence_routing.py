# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _create_wan_hybrid_evidence_collectors,
    _publish_wan_hybrid_evidence,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


def test_reversed_group_outputs_keep_request_local_evidence():
    collectors = _create_wan_hybrid_evidence_collectors(
        ["request-a", "request-b"]
    )
    collectors["request-a"].record_success(None)
    collectors["request-b"].record_success(None)
    collectors["request-b"].record_success(None)
    reversed_metrics = [RequestMetrics("request-b"), RequestMetrics("request-a")]

    _publish_wan_hybrid_evidence(reversed_metrics, collectors)

    assert reversed_metrics[0].wan_hybrid_hit_count == 2
    assert reversed_metrics[0].wan_hybrid_coverage["request_id"] == "request-b"
    assert reversed_metrics[1].wan_hybrid_hit_count == 1
    assert reversed_metrics[1].wan_hybrid_coverage["request_id"] == "request-a"


def test_duplicate_and_unmapped_request_ids_fail_closed():
    with pytest.raises(RuntimeError, match="duplicate request_id"):
        _create_wan_hybrid_evidence_collectors(["request-a", "request-a"])

    collectors = _create_wan_hybrid_evidence_collectors(["request-a"])
    with pytest.raises(RuntimeError, match="could not map output request_id"):
        _publish_wan_hybrid_evidence([RequestMetrics("request-b")], collectors)


def test_duplicate_and_missing_output_request_ids_fail_closed():
    collectors = _create_wan_hybrid_evidence_collectors(
        ["request-a", "request-b"]
    )
    with pytest.raises(RuntimeError, match="duplicate output request_id"):
        _publish_wan_hybrid_evidence(
            [RequestMetrics("request-a"), RequestMetrics("request-a")], collectors
        )
    with pytest.raises(RuntimeError, match="outputs omitted request_id"):
        _publish_wan_hybrid_evidence([RequestMetrics("request-a")], collectors)

    metrics = RequestMetrics("request-a")
    _publish_wan_hybrid_evidence(
        [metrics], collectors, require_complete=False
    )
    assert metrics.wan_hybrid_coverage["request_id"] == "request-a"
