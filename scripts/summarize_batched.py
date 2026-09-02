"""Render the checked batched-query sweep as a concise technical report."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "results" / "batched_queries_bfloat16.json"
TARGET = REPO / "docs" / "batched_queries.md"


def shape_key(row: dict) -> tuple[int, int, int, int]:
    shape = row["shape"]
    return shape["n"], shape["b"], shape["t"], shape["d"]


def main() -> None:
    data = json.loads(SOURCE.read_text())
    rows = data["batched_queries"]
    by_case = {(shape_key(row), row["n_queries"], row["impl"]): row for row in rows}

    def get(shape: tuple[int, int, int, int], queries: int, impl: str) -> dict:
        return by_case[(shape, queries, impl)]

    anchor = (9, 1, 4096, 2048)
    implementations = (
        "framework eager batched",
        "framework compiled batched",
        "switchyard x{queries} calls",
        "switchyard batched",
        "catswe phase1 S={queries}",
    )

    out: list[str] = []
    write = out.append
    write("# Batched-query forward results\n")
    write("This report uses `results/batched_queries_bfloat16.json`.")
    write("The generator is `scripts/summarize_batched.py`.\n")
    env = data["environment"]
    write(
        f"The device is {env['device_name']}. The data type is bf16. "
        "Each value is the median of 60 CUDA-event measurements."
    )
    write("The benchmark flushes the 128 MiB L2 cache before each measurement.")
    write("The benchmark excludes compilation and autotuning from steady-state latency.")
    write("Each implementation must pass the float64 oracle check before timing.\n")

    write("## Query-count sweep\n")
    write("The source shape is `N=9 B=1 T=4096 D=2048`.")
    write("The compiled baseline uses `max-autotune-no-cudagraphs`.\n")
    write("| queries | eager ms | compiled ms | separate calls ms | switchyard batched ms | catswe phase 1 ms | batched vs compiled | batched vs calls |")
    write("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for queries in (1, 2, 4, 8, 16):
        records = []
        for template in implementations:
            name = template.format(queries=queries)
            records.append(get(anchor, queries, name))
        eager, compiled, calls, switchyard, catswe = records
        sw_ms = switchyard["forward"]["median_ms"]
        write(
            f"| {queries} | {eager['forward']['median_ms']:.3f} | "
            f"{compiled['forward']['median_ms']:.3f} | "
            f"{calls['forward']['median_ms']:.3f} | {sw_ms:.3f} | "
            f"{catswe['forward']['median_ms']:.3f} | "
            f"{compiled['forward']['median_ms'] / sw_ms:.2f}x | "
            f"{calls['forward']['median_ms'] / sw_ms:.2f}x |"
        )

    write("\nThe resident switchyard path uses one kernel and no temporary workspace.")
    write("Its error is 1.00 times the bf16 rounding floor in all measured cases.")
    write("The catswe path also has rounding-floor accuracy.")
    write("Catswe also computes merge statistics and supports its backward design.")
    write("The switchyard batched API returns only the output and has no backward.\n")

    write("## Shape checks at eight queries\n")
    write("| shape | compiled ms | switchyard ms | catswe ms | switchyard kernels | switchyard workspace MiB |")
    write("|---|---:|---:|---:|---:|---:|")
    shapes = (
        (2, 1, 4096, 2048),
        (16, 1, 4096, 2048),
        (9, 1, 4096, 1024),
        (9, 1, 128, 1024),
        (9, 4, 2048, 2048),
        (9, 1, 4096, 4096),
    )
    for shape in shapes:
        compiled = get(shape, 8, "framework compiled batched")
        switchyard = get(shape, 8, "switchyard batched")
        catswe = get(shape, 8, "catswe phase1 S=8")
        shape_text = f"N={shape[0]} B={shape[1]} T={shape[2]} D={shape[3]}"
        write(
            f"| {shape_text} | {compiled['forward']['median_ms']:.3f} | "
            f"{switchyard['forward']['median_ms']:.3f} | "
            f"{catswe['forward']['median_ms']:.3f} | "
            f"{switchyard['forward_kernels']['total_kernels']:.0f} | "
            f"{switchyard['forward_memory']['workspace_bytes'] / 2**20:.3f} |"
        )

    write("\nThe `D=4096` case exceeds the resident tile budget.")
    write("The fallback uses eight per-query kernels and one stack kernel.")
    write("It takes 2.241 ms and uses 256 MiB of temporary workspace.")
    write("Catswe takes 0.457 ms in this case.")
    write("Use this batched API only in its documented resident dispatch range.\n")

    write("## Reproduction\n")
    write("```bash")
    write("source scripts/env.sh")
    write("scripts/fetch_third_party.sh")
    write("python bench/bench_third_party.py --batched-only --dtype bfloat16")
    write("python scripts/summarize_batched.py")
    write("```")

    TARGET.write_text("\n".join(out) + "\n")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
