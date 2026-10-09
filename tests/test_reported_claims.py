"""CPU checks for historical claim arithmetic and its limits."""

import importlib.util
import json
from pathlib import Path
from shutil import copyfile

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_reported_claims", REPO / "scripts/check_reported_claims.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_historical_claims_keep_denominators_distinct():
    audit = MODULE.check_claims(REPO / "results")
    framework = audit["model"]["attnres/arena"]
    fused = audit["model"]["switchyard/arena"]
    assert framework["params"] == fused["params"] == 1_300_535_296
    assert framework["control_difference_share_pct"] == pytest.approx(39.1, abs=0.05)
    assert fused["control_difference_share_pct"] == pytest.approx(11.4, abs=0.05)
    assert fused["overhead_relative_to_control_pct"] == pytest.approx(12.9, abs=0.05)
    assert fused["profiler_device_time_share_pct"] == pytest.approx(7.4, abs=0.05)
    assert audit["model_step_time_reduction_pct"] == pytest.approx(31.3, abs=0.05)
    assert audit["model_throughput_speedup"] == pytest.approx(1.46, abs=0.005)
    assert audit["operator_speedups"]["forward"] == pytest.approx(1.70, abs=0.005)
    assert audit["operator_speedups"]["fwd_bwd"] == pytest.approx(3.71, abs=0.005)
    assert audit["batched_logical_min_effective_tbps"] == pytest.approx(1.436, abs=0.0005)
    assert audit["artifacts"]["model_bfloat16.json"]["recorded_provenance"] is None


@pytest.mark.parametrize("field", ["tokens_per_second", "step_share_pct", "logical_min_bytes"])
def test_inconsistent_claim_artifacts_fail(tmp_path, field):
    for name in ("model_bfloat16", "operator_representative_bfloat16", "batched_queries_bfloat16"):
        copyfile(REPO / "results" / f"{name}.json", tmp_path / f"{name}.json")
    name = "batched_queries_bfloat16" if field == "logical_min_bytes" else "model_bfloat16"
    path = tmp_path / f"{name}.json"
    data = json.loads(path.read_text())
    if field == "tokens_per_second":
        data["variants"][0][field] *= 2
    elif field == "step_share_pct":
        data["attribution"][1][field] *= 2
    else:
        for row in data["batched_queries"]:
            row[field] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inconsistent"):
        MODULE.check_claims(tmp_path)
