"""Turn results/operator_*.json into docs/results.md and the plots it references.

Nothing in the documentation is typed by hand. Run this after
``bench/bench_operator.py`` and the tables and figures follow the data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DOCS = REPO / "docs"

#: Display order and short labels. Anything not listed is still reported, at the end.
LABELS = {
    "speed_of_light": "speed of light",
    "paper_eager": "eager, paper form",
    "folded_eager": "eager, folded form",
    "paper_compiled": "compile, paper form",
    "folded_compiled": "compile, folded form",
    "folded_compiled_cudagraph": "compile + cudagraph",
    "switchyard_triton": "**switchyard**",
}
BASELINES = ("paper_eager", "folded_eager", "paper_compiled",
             "folded_compiled", "folded_compiled_cudagraph")


def load() -> tuple[list[dict], dict]:
    rows, env = [], {}
    for path in sorted(RESULTS.glob("operator_*.json")):
        data = json.loads(path.read_text())
        env = data.get("environment", env)
        for r in data["results"]:
            r["_set"] = data.get("shape_set")
            r["_dtype"] = data.get("dtype")
            rows.append(r)
    return rows, env


def shape_key(s: dict) -> str:
    return f"N={s['n']} B={s['b']} T={s['t']} D={s['d']}"


def by_shape(rows: list[dict]) -> dict:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        # Later files win on duplicate (shape, impl); sets overlap deliberately.
        out[shape_key(r["shape"])][r["impl"]] = r
    return out


def fmt_ms(r: dict | None) -> str:
    if not r or r.get("skipped") or "median_ms" not in r.get("forward", {}):
        return "--"
    return f"{r['forward']['median_ms']:.3f}"


def best_baseline(impls: dict, field: str) -> tuple[str, float] | None:
    """Fastest non-switchyard, non-ceiling implementation for a given metric."""
    best = None
    for name in BASELINES:
        r = impls.get(name)
        if not r or r.get("skipped"):
            continue
        block = r.get(field, {})
        v = block.get("median_ms")
        if v is None:
            continue
        if best is None or v < best[1]:
            best = (name, v)
    return best


def table_forward(grouped: dict) -> list[str]:
    out = ["| shape | " + " | ".join(LABELS.values()) + " | ours vs best baseline | % of ceiling |",
           "|---" * (len(LABELS) + 3) + "|"]
    for shape, impls in grouped.items():
        cells = [fmt_ms(impls.get(k)) for k in LABELS]
        ours = impls.get("switchyard_triton")
        bb = best_baseline(impls, "forward")
        speedup = "--"
        if ours and not ours.get("skipped") and bb:
            speedup = f"**{bb[1] / ours['forward']['median_ms']:.2f}x**"
        sol = "--"
        if ours and ours.get("fraction_of_speed_of_light"):
            sol = f"{100 * ours['fraction_of_speed_of_light']:.0f}%"
        out.append(f"| {shape} | " + " | ".join(cells) + f" | {speedup} | {sol} |")
    return out


def table_fwd_bwd(grouped: dict) -> list[str]:
    cols = [k for k in LABELS if k != "speed_of_light"]
    out = ["| shape | " + " | ".join(LABELS[k] for k in cols) + " | ours vs best baseline |",
           "|---" * (len(cols) + 2) + "|"]
    for shape, impls in grouped.items():
        cells = []
        for k in cols:
            r = impls.get(k)
            fb = (r or {}).get("fwd_bwd", {})
            cells.append(f"{fb['median_ms']:.3f}" if "median_ms" in fb else "--")
        ours = impls.get("switchyard_triton")
        bb = best_baseline(impls, "fwd_bwd")
        sp = "--"
        if ours and "median_ms" in ours.get("fwd_bwd", {}) and bb:
            ratio = bb[1] / ours["fwd_bwd"]["median_ms"]
            sp = f"**{ratio:.2f}x**" if ratio >= 1.0 else f"{ratio:.2f}x"
        out.append(f"| {shape} | " + " | ".join(cells) + f" | {sp} |")
    return out


def table_kernels_and_memory(grouped: dict) -> list[str]:
    out = ["| shape | impl | fwd kernels | fwd+bwd kernels | workspace MiB | GPU busy % |",
           "|---|---|---|---|---|---|"]
    for shape, impls in grouped.items():
        for name in ("paper_compiled", "folded_compiled", "switchyard_triton"):
            r = impls.get(name)
            if not r or r.get("skipped"):
                continue
            fk = r.get("forward_kernels", {}).get("total_kernels", "--")
            bk = r.get("fwd_bwd_kernels", {}).get("total_kernels", "--")
            ws = r.get("forward_memory", {}).get("workspace_bytes")
            wss = f"{ws / 2**20:.1f}" if ws is not None else "--"
            util = r.get("forward_gpu_utilization")
            us = f"{100 * util:.0f}%" if util is not None else "--"
            out.append(f"| {shape} | {LABELS[name]} | {fk} | {bk} | {wss} | {us} |")
    return out


def plots(grouped: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    DOCS.mkdir(exist_ok=True)
    made = []

    # Achieved bandwidth against the ceiling, over the N sweep.
    ns, series = [], defaultdict(list)
    for shape, impls in sorted(
        grouped.items(), key=lambda kv: int(kv[0].split()[0].split("=")[1])
    ):
        parts = dict(p.split("=") for p in shape.split())
        if (int(parts["B"]), int(parts["T"]), int(parts["D"])) != (1, 4096, 2048):
            continue
        ns.append(int(parts["N"]))
        for name in ("speed_of_light", "paper_compiled", "folded_compiled", "switchyard_triton"):
            r = impls.get(name)
            series[name].append(
                r["forward_achieved_gbps"] if r and not r.get("skipped")
                and "forward_achieved_gbps" in r else float("nan")
            )
    if ns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        styles = {
            "speed_of_light": dict(color="0.4", ls="--", marker="", label="speed of light"),
            "paper_compiled": dict(color="tab:orange", marker="s", label="torch.compile (paper form)"),
            "folded_compiled": dict(color="tab:green", marker="^", label="torch.compile (folded form)"),
            "switchyard_triton": dict(color="tab:blue", marker="o", lw=2.2, label="switchyard"),
        }
        for name, st in styles.items():
            if name in series:
                ax.plot(ns, series[name], **st)
        ax.set_xlabel("N  (number of source states)")
        ax.set_ylabel("achieved bandwidth (GB/s)")
        ax.set_title("Block AttnRes forward, bf16, B=1 T=4096 D=2048\nRTX PRO 6000 Blackwell")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(DOCS / "bandwidth_vs_n.png", dpi=150)
        plt.close(fig)
        made.append("bandwidth_vs_n.png")
    return made


def main() -> None:
    rows, env = load()
    if not rows:
        raise SystemExit("no results/operator_*.json found; run bench/bench_operator.py first")
    grouped = by_shape(rows)

    out: list[str] = []
    w = out.append
    w("# Operator results\n")
    w("Generated by `scripts/summarize_operator.py` from `results/operator_*.json`.")
    w("Every figure below comes from those files; none is typed by hand.\n")
    w("**Measurement.** Median of 100 CUDA-event-timed repetitions after 25 warmup")
    w("iterations, with the 128 MiB L2 flushed between repetitions. Compilation and")
    w("autotuning happen before timing and are reported separately in the JSON. Every")
    w("implementation is checked against a float64 oracle before it is timed; anything")
    w("that fails is excluded rather than reported.\n")
    w("**Speed of light** is a Triton kernel that touches exactly the operator's bytes")
    w("with the same access pattern, tile shape and launch geometry, but skips the")
    w("softmax. No correct implementation can beat it, which makes it a fairer ceiling")
    w("than a generic copy benchmark.\n")
    if env:
        w(f"**Environment.** {env.get('device_name')} (CC {env.get('device_cc')}), "
          f"torch {env.get('torch')}+cu{env.get('torch_cuda')}, triton {env.get('triton')}, "
          f"bf16.\n")

    w("## Forward latency (ms)\n")
    out.extend(table_forward(grouped))
    w("")
    w("## Forward + backward latency (ms)\n")
    out.extend(table_fwd_bwd(grouped))
    w("")
    w("## Kernel counts and workspace\n")
    w("Workspace is peak allocator high-water mark minus the bytes the caller already")
    w("holds (its sources plus the output). It is the memory an implementation needs")
    w("*on top of* the data, which is what fusion removes. The sources themselves are")
    w("a property of the architecture and do not go away.\n")
    out.extend(table_kernels_and_memory(grouped))
    w("")

    made = plots(grouped)
    if made:
        w("## Figures\n")
        for m in made:
            w(f"![{m}]({m})\n")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "results.md").write_text("\n".join(out) + "\n")
    print(f"wrote {DOCS / 'results.md'}")
    for m in made:
        print(f"wrote {DOCS / m}")


if __name__ == "__main__":
    main()
