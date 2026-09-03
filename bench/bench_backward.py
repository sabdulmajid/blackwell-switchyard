"""Compare complete backward architectures before any dispatch change.

This benchmark is intentionally separate from the accepted operator sweep. The
candidates are private and normal dispatch cannot select them. A result from
this file is evidence for a dispatch change, not a dispatch change by itself.

Every timed implementation first runs forward and backward against a float64
oracle on the same shape. Latency uses CUDA events with an L2 flush between
repetitions. Compilation and warmup are outside the timed region.

Run after the target GPU is available::

    source scripts/env.sh
    python bench/bench_backward.py --shape-set gate --dtype bfloat16
    python scripts/evaluate_backward.py \
      results/backward_candidates_bfloat16.json \
      results/backward_candidates_float16.json \
      --candidate cuda_cluster
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY_DIR", REPO / "third_party"))
liger_src = THIRD_PARTY / "Liger-Kernel" / "src"
if liger_src.exists():
    sys.path.insert(0, str(liger_src))

from harness import (  # noqa: E402
    check_against,
    count_kernels,
    environment,
    measure_latency,
    measure_memory,
    repository_provenance,
)
from switchyard.performance import backward_traffic_estimate  # noqa: E402
from switchyard.reference import (  # noqa: E402
    DEFAULT_EPS,
    block_attn_res_reference,
)
from switchyard.training_plan import get_training_plan, plan_supports  # noqa: E402
from switchyard.triton_op import (  # noqa: E402
    _block_attn_res_source_serial,
    _block_attn_res_with_plan,
    _bwd_launch,
    block_attn_res_triton,
)


@dataclass(frozen=True)
class Shape:
    n: int
    b: int
    t: int
    d: int

    def key(self) -> str:
        return f"N{self.n}_B{self.b}_T{self.t}_D{self.d}"


# ``gate`` targets every regime where the stored head-to-head says Liger wins.
# ``full`` adds dispatch-boundary controls, token-count crossovers, and a
# batched training layout. It is bounded; it is not a Cartesian product.
SHAPE_SETS = {
    "gate": [
        Shape(9, 1, 4096, 4096),
        Shape(9, 1, 4096, 8192),
        Shape(32, 1, 4096, 2048),
        Shape(32, 1, 4096, 4096),
    ],
    "full": [
        Shape(8, 1, 4096, 4096),
        Shape(9, 1, 128, 4096),
        Shape(9, 1, 512, 4096),
        Shape(9, 1, 4096, 4096),
        Shape(9, 1, 8192, 4096),
        Shape(4, 1, 4096, 8192),
        Shape(5, 1, 4096, 8192),
        Shape(9, 1, 4096, 8192),
        Shape(16, 1, 4096, 2048),
        Shape(17, 1, 4096, 2048),
        Shape(32, 1, 4096, 2048),
        Shape(9, 4, 2048, 4096),
    ],
}

CORRECTNESS_ONLY_SHAPES = [
    Shape(9, 1, 129, 4097),
    Shape(17, 2, 33, 2049),
]

TOLERANCES = {
    torch.bfloat16: {"output": 2e-2, "dv": 3e-2, "dw": 6e-2},
    torch.float16: {"output": 5e-3, "dv": 1e-2, "dw": 3e-2},
    torch.float32: {"output": 2e-5, "dv": 1e-4, "dw": 1e-3},
}


def _compute_processes(device_uuid: str) -> tuple[list[str], list[str]]:
    """Return this benchmark's CUDA contexts and all other GPU processes."""
    process_query = [
        "nvidia-smi",
        f"--id={device_uuid}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.run(
        process_query, check=True, capture_output=True, text=True
    ).stdout.strip().splitlines()
    rows = [row for row in rows if row.strip()]
    own_pid = str(os.getpid())
    own_context = [row for row in rows if row.split(",", 1)[0].strip() == own_pid]
    return own_context, [row for row in rows if row not in own_context]


def _gpu_preflight(device: torch.device, *, allow_busy: bool) -> dict:
    """Record device state and refuse to measure beside another process."""
    properties = torch.cuda.get_device_properties(device)
    device_uuid = properties.uuid
    query = [
        "nvidia-smi",
        f"--id={device_uuid}",
        "--query-gpu=uuid,name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    state = subprocess.run(query, check=True, capture_output=True, text=True).stdout.strip()
    own_context, other_processes = _compute_processes(device_uuid)
    if other_processes and not allow_busy:
        raise SystemExit(
            "selected GPU has active compute processes; stop and rerun only with exclusive access: "
            + "; ".join(other_processes)
        )
    return {
        "logical_device": str(device),
        "resolved_uuid": device_uuid,
        "resolved_pci_bus_id": properties.pci_bus_id,
        "device_query": state,
        "benchmark_process_context": own_context,
        "compute_processes_at_start": other_processes,
        "exclusive_access_required": True,
        "busy_override": allow_busy,
    }


def _gpu_postflight(preflight: dict) -> dict:
    """Confirm that no competing process appeared during the benchmark."""
    own_context, other_processes = _compute_processes(preflight["resolved_uuid"])
    return {
        "resolved_uuid": preflight["resolved_uuid"],
        "benchmark_process_context": own_context,
        "compute_processes_at_end": other_processes,
    }


def _oracle(v: torch.Tensor, w: torch.Tensor, g: torch.Tensor):
    """Return float64 output and first-order gradients on the input device."""
    v64 = v.to(torch.float64).requires_grad_(True)
    w64 = w.to(torch.float64).requires_grad_(True)
    out64 = block_attn_res_reference(v64, w64, DEFAULT_EPS)
    dv64, dw64 = torch.autograd.grad(out64, (v64, w64), g.to(torch.float64))
    return out64.detach(), dv64.detach(), dw64.detach()


def _correctness(fn, v, w, g, oracle, tolerances) -> dict:
    """Check the complete training contract before any timing call."""
    v_test = v.detach().clone().requires_grad_(True)
    w_test = w.detach().clone().requires_grad_(True)
    out = fn(v_test, w_test, DEFAULT_EPS)
    dv, dw = torch.autograd.grad(out, (v_test, w_test), g)
    out64, dv64, dw64 = oracle
    return {
        "output": check_against(lambda: out, out64, (), rel_l2_tol=tolerances["output"]),
        "dv": check_against(lambda: dv, dv64, (), rel_l2_tol=tolerances["dv"]),
        "dw": check_against(lambda: dw, dw64, (), rel_l2_tol=tolerances["dw"]),
    }


def _correct(report: dict) -> bool:
    return all(report[name].get("ok", False) for name in ("output", "dv", "dw"))


def _summarize_trials(trials: list[dict], *, warmup: int) -> dict:
    """Summarize raw CUDA-event samples while retaining trial boundaries."""
    samples = [sample for trial in trials for sample in trial["samples_ms"]]
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    return {
        "median_ms": statistics.median(samples),
        "trial_medians_ms": [statistics.median(trial["samples_ms"]) for trial in trials],
        "p10_ms": ordered[int(0.10 * len(ordered))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "min_ms": ordered[0],
        "mean_ms": mean,
        "cv": statistics.pstdev(samples) / mean if mean > 0 else 0.0,
        "reps": len(samples),
        "warmup_per_trial": warmup,
        "trial_count": len(trials),
        "l2_flushed": True,
        "trials": trials,
    }


def _measure_paired_trials(
    functions: dict[str, Callable[[], object]],
    *,
    device: torch.device,
    quick: bool,
    rotation: int,
) -> tuple[dict[str, dict], list[dict]]:
    """Interleave implementations and preserve paired trial medians.

    CUDA-event samples within one trial are autocorrelated. The evaluator
    therefore treats the interleaved trial medians, not every repetition, as
    the independent paired observations.
    """
    trial_count, reps, warmup = (2, 10, 8) if quick else (5, 40, 25)
    names = list(functions)
    trials = {name: [] for name in names}
    schedule = []
    for trial_index in range(trial_count):
        offset = (rotation + trial_index) % len(names)
        order = names[offset:] + names[:offset]
        schedule.append({"trial": trial_index, "implementations": order})
        for order_index, name in enumerate(order):
            measured = measure_latency(
                functions[name],
                device=device,
                warmup=warmup,
                reps=reps,
                record_samples=True,
            ).as_dict()
            measured["trial"] = trial_index
            measured["order_in_trial"] = order_index
            trials[name].append(measured)
    return (
        {name: _summarize_trials(values, warmup=warmup) for name, values in trials.items()},
        schedule,
    )


def build_implementations(v: torch.Tensor) -> tuple[dict, list[str]]:
    """Return accepted, candidate, and optional Liger implementations."""
    def plan_spec(name: str, status: str) -> dict:
        plan = get_training_plan(name)

        def run(values, query, eps=DEFAULT_EPS):
            return _block_attn_res_with_plan(values, query, eps, plan_name=name)

        return {"fn": run, "status": status, "plan": plan}

    implementations = {
        "current": {
            "fn": block_attn_res_triton,
            "status": "accepted production dispatch",
        },
        "source_serial": {
            "fn": _block_attn_res_source_serial,
            "status": "private candidate; not reachable from production dispatch",
            "plan": get_training_plan("serial_recompute_atomic_t1"),
        },
        "serial_recompute_atomic_t4": plan_spec(
            "serial_recompute_atomic_t4",
            "private grouped L2-local control",
        ),
        "serial_saved_atomic_t4": plan_spec(
            "serial_saved_atomic_t4",
            "private grouped saved-state control",
        ),
        "serial_saved_partials_t16": plan_spec(
            "serial_saved_partials_t16",
            "private saved-state hierarchical-reduction candidate",
        ),
        "cuda_shared": plan_spec(
            "cuda_shared",
            "private one-token one-read shared-memory control",
        ),
        "cuda_cluster": plan_spec(
            "cuda_cluster",
            "private persistent feature-sharded one-read cluster candidate",
        ),
    }
    notes: list[str] = []
    try:
        from liger_kernel.ops.attn_res import LigerAttnResFunction

        gain = torch.ones(v.shape[-1], device=v.device, dtype=v.dtype)

        def liger(values, query, eps=DEFAULT_EPS):
            return LigerAttnResFunction.apply(values, query, gain, eps)

        implementations["liger"] = {
            "fn": liger,
            "status": "pinned third-party comparator; RMSNorm gain fixed to one",
        }
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Liger unavailable: {type(exc).__name__}: {exc}"[:300])
    return implementations, notes


def _make_runtime(fn, v: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> dict:
    """Build equivalent forward, backward, and training-step callables."""
    vg = v.detach().clone().requires_grad_(True)
    wg = w.detach().clone().requires_grad_(True)
    retained_out = fn(vg, wg, DEFAULT_EPS)

    def backward_only():
        vg.grad = None
        wg.grad = None
        retained_out.backward(g, retain_graph=True)

    vg_step = v.detach().clone().requires_grad_(True)
    wg_step = w.detach().clone().requires_grad_(True)

    def fwd_bwd():
        vg_step.grad = None
        wg_step.grad = None
        fn(vg_step, wg_step, DEFAULT_EPS).backward(g)

    def clear_gradients():
        vg_step.grad = None
        wg_step.grad = None

    return {
        "forward": lambda: fn(v, w, DEFAULT_EPS),
        "backward": backward_only,
        "fwd_bwd": fwd_bwd,
        "clear_gradients": clear_gradients,
    }


def bench_one(
    name,
    spec,
    shape,
    v,
    w,
    g,
    correctness_by_seed,
    device,
    runtime,
    timings,
) -> dict:
    """Add profiles and memory after paired timing and correctness gates."""
    record = {
        "impl": name,
        "shape": asdict(shape),
        "status": spec["status"],
        "correctness": correctness_by_seed[0]["report"],
        "correctness_by_seed": correctness_by_seed,
    }
    plan = spec.get("plan")
    if plan is not None:
        record["training_plan"] = plan.as_dict()
    if not all(_correct(item["report"]) for item in correctness_by_seed):
        record["skipped"] = "failed output or gradient correctness for at least one seed"
        return record
    record.update(timings)

    try:
        record["backward_kernels"] = count_kernels(
            runtime["backward"], device=device, iters=3
        ).as_dict()
    except Exception as exc:  # noqa: BLE001
        record["backward_kernels"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    try:
        record["fwd_bwd_kernels"] = count_kernels(
            runtime["fwd_bwd"], device=device, iters=3
        ).as_dict()
    except Exception as exc:  # noqa: BLE001
        record["fwd_bwd_kernels"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    itemsize = v.element_size()
    resident = (
        v.numel() * itemsize
        + w.numel() * itemsize
        + g.numel() * itemsize
        + v.numel() * itemsize
        + w.numel() * itemsize
    )
    record["fwd_bwd_memory"] = measure_memory(
        runtime["fwd_bwd"],
        device=device,
        resident_bytes=resident,
        output_bytes=(shape.b * shape.t * shape.d + v.numel() + w.numel()) * itemsize,
        after_warmup=runtime["clear_gradients"],
    ).as_dict()

    if name == "current":
        n_pow2 = 1 << (shape.n - 1).bit_length()
        current_is_resident = _bwd_launch(n_pow2, shape.d)[0]
        model_name = "resident" if current_is_resident else "tiled"
        record["traffic_model"] = backward_traffic_estimate(
            model_name,
            shape.n,
            shape.b,
            shape.t,
            shape.d,
            itemsize=itemsize,
        ).as_dict()
    elif plan is not None:
        family = plan.backward.family
        traffic_options = {}
        if family == "cuda_cluster":
            from switchyard.cuda_op import cuda_cluster_launch_info

            launch_info = cuda_cluster_launch_info(v)
            record["cluster_launch_info"] = launch_info
            traffic_options["persistent_clusters"] = min(
                shape.b * shape.t, launch_info["active_clusters"]
            )
        model_name = {
            "source_serial": (
                "source_serial_saved" if plan.saves_forward_stats else "source_serial"
            ),
            "cuda_shared": "cuda_shared",
            "cuda_cluster": "cuda_cluster",
        }[family]
        record["traffic_model"] = backward_traffic_estimate(
            model_name,
            shape.n,
            shape.b,
            shape.t,
            shape.d,
            itemsize=itemsize,
            source_tokens_per_cta=plan.backward.tokens_per_cta,
            source_uses_partials=plan.backward.dw_reduction == "partials",
            **traffic_options,
        ).as_dict()

    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shape-set", choices=sorted(SHAPE_SETS), default="gate")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--impls",
        default=(
            "current,serial_recompute_atomic_t4,serial_saved_partials_t16,"
            "cuda_shared,cuda_cluster,liger"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--correctness-seeds", default="0,1,2")
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="diagnostic use only; results cannot pass the production gate",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    preflight = _gpu_preflight(device, allow_busy=args.allow_busy_gpu)
    dtype = getattr(torch, args.dtype)
    selected = [name.strip() for name in args.impls.split(",") if name.strip()]
    correctness_seeds = [
        int(value.strip()) for value in args.correctness_seeds.split(",") if value.strip()
    ]
    if not correctness_seeds or correctness_seeds[0] != 0:
        parser.error("--correctness-seeds must start with 0, the timed input seed")

    report = {
        "schema_version": 2,
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "experiment": "backward architecture selection",
        "candidate_reachable_from_production": False,
        "environment": environment(),
        "gpu_preflight": preflight,
        "provenance": repository_provenance(
            REPO, {"Liger-Kernel": THIRD_PARTY / "Liger-Kernel"}
        ),
        "dtype": args.dtype,
        "shape_set": args.shape_set,
        "tolerances": TOLERANCES[dtype],
        "correctness_seeds": correctness_seeds,
        "methodology": {
            "oracle": "float64 forward and first-order gradients on each timed shape",
            "timing": (
                "five interleaved trials of 40 CUDA-event samples; all raw samples and "
                "trial order stored"
            ),
            "statistics": (
                "paired trial medians are independent observations; individual event "
                "samples are not treated as independent"
            ),
            "cache": "L2 flushed after graph setup and before every timed region",
            "compilation": "excluded by warmup",
            "backward": "measured directly on a retained graph; not median subtraction",
            "promotion": "run scripts/evaluate_backward.py on this raw result",
        },
        "notes": [],
        "results": [],
        "correctness_only": [],
        "execution_order": [],
    }

    for shape_index, shape in enumerate(SHAPE_SETS[args.shape_set]):
        print(shape.key(), flush=True)
        torch.manual_seed(0)
        v = torch.randn(shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype)
        w = torch.randn(shape.d, device=device, dtype=torch.float32)
        w = (w / w.norm()).to(dtype)
        torch.manual_seed(1)
        g = torch.randn(shape.b, shape.t, shape.d, device=device, dtype=dtype)

        implementations, notes = build_implementations(v)
        report["notes"].extend(note for note in notes if note not in report["notes"])
        unknown = sorted(set(selected) - set(implementations))
        if unknown:
            report["notes"].append(f"unavailable requested implementations: {unknown}")

        supported: dict[str, tuple[bool, str]] = {}
        for name in selected:
            spec = implementations.get(name)
            if spec is None or spec.get("plan") is None:
                supported[name] = (spec is not None, "framework implementation")
            else:
                supported[name] = plan_supports(
                    spec["plan"],
                    shape.n,
                    shape.b,
                    shape.t,
                    shape.d,
                    args.dtype,
                )
                if not supported[name][0]:
                    report["results"].append(
                        {
                            "impl": name,
                            "shape": asdict(shape),
                            "status": spec["status"],
                            "training_plan": spec["plan"].as_dict(),
                            "skipped": f"unsupported plan: {supported[name][1]}",
                        }
                    )

        correctness: dict[str, list[dict]] = {
            name: [] for name in selected if supported.get(name, (False, ""))[0]
        }
        for seed in correctness_seeds:
            if seed == 0:
                seed_v, seed_w, seed_g = v, w, g
            else:
                torch.manual_seed(seed)
                seed_v = torch.randn(
                    shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype
                )
                seed_w = torch.randn(shape.d, device=device, dtype=torch.float32)
                seed_w = (seed_w / seed_w.norm()).to(dtype)
                torch.manual_seed(1000 + seed)
                seed_g = torch.randn(shape.b, shape.t, shape.d, device=device, dtype=dtype)
            seed_oracle = _oracle(seed_v, seed_w, seed_g)
            for name in selected:
                if name in correctness:
                    correctness[name].append(
                        {
                            "seed": seed,
                            "report": _correctness(
                                implementations[name]["fn"],
                                seed_v,
                                seed_w,
                                seed_g,
                                seed_oracle,
                                TOLERANCES[dtype],
                            ),
                        }
                    )
            del seed_oracle
            if seed != 0:
                del seed_v, seed_w, seed_g
                torch.cuda.empty_cache()

        available_order = [name for name in selected if name in correctness]
        if not available_order:
            raise SystemExit("none of the requested implementations is available")
        offset = shape_index % len(available_order)
        available_order = available_order[offset:] + available_order[:offset]
        valid_order = [
            name
            for name in available_order
            if all(_correct(item["report"]) for item in correctness[name])
        ]
        runtimes = {
            name: _make_runtime(implementations[name]["fn"], v, w, g)
            for name in valid_order
        }
        timings = {name: {} for name in valid_order}
        shape_schedule = {"shape": asdict(shape), "metrics": {}}
        for metric_index, metric in enumerate(("forward", "backward", "fwd_bwd")):
            if not valid_order:
                break
            measured, schedule = _measure_paired_trials(
                {name: runtimes[name][metric] for name in valid_order},
                device=device,
                quick=args.quick,
                rotation=shape_index + metric_index,
            )
            shape_schedule["metrics"][metric] = schedule
            for name in valid_order:
                timings[name][metric] = measured[name]
        report["execution_order"].append(shape_schedule)

        for name in available_order:
            print(f"  {name}", flush=True)
            report["results"].append(
                bench_one(
                    name,
                    implementations[name],
                    shape,
                    v,
                    w,
                    g,
                    correctness[name],
                    device,
                    runtimes.get(name, {}),
                    timings.get(name, {}),
                )
            )

        del v, w, g
        torch.cuda.empty_cache()

    for shape in CORRECTNESS_ONLY_SHAPES:
        print(f"correctness-only {shape.key()}", flush=True)
        case = {"shape": asdict(shape), "implementations": []}
        for seed in correctness_seeds:
            torch.manual_seed(seed)
            v = torch.randn(shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype)
            w = torch.randn(shape.d, device=device, dtype=torch.float32)
            w = (w / w.norm()).to(dtype)
            torch.manual_seed(1000 + seed)
            g = torch.randn(shape.b, shape.t, shape.d, device=device, dtype=dtype)
            oracle = _oracle(v, w, g)
            implementations, _ = build_implementations(v)
            for name in selected:
                spec = implementations.get(name)
                if spec is not None and spec.get("plan") is not None:
                    is_supported, reason = plan_supports(
                        spec["plan"],
                        shape.n,
                        shape.b,
                        shape.t,
                        shape.d,
                        args.dtype,
                    )
                    if not is_supported:
                        case["implementations"].append(
                            {
                                "impl": name,
                                "seed": seed,
                                "training_plan": spec["plan"].as_dict(),
                                "skipped": f"unsupported plan: {reason}",
                            }
                        )
                        continue
                if spec is not None:
                    case["implementations"].append(
                        {
                            "impl": name,
                            "seed": seed,
                            "correctness": _correctness(
                                spec["fn"],
                                v,
                                w,
                                g,
                                oracle,
                                TOLERANCES[dtype],
                            ),
                        }
                    )
            del oracle, v, w, g
            torch.cuda.empty_cache()
        report["correctness_only"].append(case)

    report["gpu_postflight"] = _gpu_postflight(preflight)
    out = args.out or REPO / "results" / f"backward_candidates_{args.dtype}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
