"""Recompute historical README claims without running or validating GPU code."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def check_claims(results: Path) -> dict:
    reports = {}
    artifacts = {}
    for name in ("model_bfloat16", "operator_representative_bfloat16", "batched_queries_bfloat16"):
        path = results / f"{name}.json"
        raw = path.read_bytes()
        report = reports[name] = json.loads(raw)
        if report["dtype"] != "bfloat16":
            raise ValueError(f"{name}: expected bfloat16")
        artifacts[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "environment": report["environment"],
            "recorded_provenance": report.get("provenance"),
        }

    model = reports["model_bfloat16"]
    variants = {v["variant"]: v for v in model["variants"]}
    control, framework, fused = (variants[n] for n in (
        "standard", "attnres/arena", "switchyard/arena"
    ))
    if framework["params"] != fused["params"]:
        raise ValueError("AttnRes variants have different parameter counts")
    tokens = model["batch"] * model["seq"]
    attribution = {a["variant"]: a for a in model["attribution"]}
    model_rows = {}
    c = control["step"]["median_ms"]
    for v in (control, framework, fused):
        step = v["step"]["median_ms"]
        throughput = tokens * 1000 / step
        if not math.isclose(throughput, v["tokens_per_second"], rel_tol=1e-9):
            raise ValueError(f"{v['variant']}: inconsistent tokens/s")
        share = 100 * (step - c) / step
        if v is not control and not math.isclose(
            share, attribution[v["variant"]]["step_share_pct"], rel_tol=1e-9
        ):
            raise ValueError(f"{v['variant']}: inconsistent control-difference share")
        profile_share = v["profile"]["attnres_share_of_device_time"]
        model_rows[v["variant"]] = {
            "params": v["params"]["total"],
            "step_ms": step,
            "tokens_per_second": throughput,
            "control_difference_ms": step - c,
            "control_difference_share_pct": share,
            "overhead_relative_to_control_pct": 100 * (step - c) / c,
            "profiler_device_time_share_pct": (
                None if profile_share is None else 100 * profile_share
            ),
        }

    shape = {"n": 9, "b": 1, "t": 4096, "d": 2048}
    op = {r["impl"]: r for r in reports["operator_representative_bfloat16"]["results"]
          if r["shape"] == shape}
    baseline, custom = op["folded_compiled_autotune"], op["switchyard_triton"]
    batched = next(r for r in reports["batched_queries_bfloat16"]["batched_queries"]
                   if r["impl"] == "switchyard batched" and r["shape"] == shape
                   and r["n_queries"] == 8)
    for row in (baseline, custom, batched):
        if row["correctness"]["ok"] is not True:
            raise ValueError(f"{row['impl']}: recorded correctness failed")
    # One source read, one query read, and one output write; not measured DRAM traffic.
    logical_bytes = ((shape["n"] + 8) * shape["b"] * shape["t"] * shape["d"]
                     + 8 * shape["d"]) * 2
    gbps = logical_bytes / (batched["forward"]["median_ms"] * 1e6)
    if logical_bytes != batched["logical_min_bytes"] or not math.isclose(
        gbps, batched["logical_min_achieved_gbps"], rel_tol=1e-9
    ):
        raise ValueError("batched logical-minimum bandwidth is inconsistent")
    f, s = (v["step"]["median_ms"] for v in (framework, fused))
    return {
        "scope": "Historical arithmetic audit only; not a new GPU or release validation",
        "artifacts": artifacts,
        "model_configuration": {"config": model["config"], "batch": model["batch"],
                                "seq": model["seq"], "tokens_per_step": tokens},
        "operator_shape": shape,
        "batched_query_count": 8,
        "model": model_rows,
        "model_throughput_speedup": f / s,
        "model_step_time_reduction_pct": 100 * (f - s) / f,
        "operator_speedups": {m: baseline[m]["median_ms"] / custom[m]["median_ms"]
                              for m in ("forward", "fwd_bwd")},
        "batched_logical_min_bytes": logical_bytes,
        "batched_logical_min_effective_tbps": gbps / 1000,
    }


if __name__ == "__main__":
    print(json.dumps(check_claims(REPO / "results"), indent=2))
