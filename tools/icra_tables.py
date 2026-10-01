#!/usr/bin/env python3
"""Print the docs/icra-replication.md result tables from a `tools/timing.py run` ICRA directory.

Usage: icra_tables.py ICRA_RUN_DIR
Reads the harness verdicts and raw per-solve samples; measures nothing. The statistics match
tools/plot_timing.py: Figure 4 and the Figure 6 grid are means over repeats, Figure 5 pools the
repeats' raw solves. Published values come from the paper's text and Figure 6 table.
"""
import json
from pathlib import Path
import sys

import numpy as np

HORIZONS = [32, 64, 128, 256, 512]
PUBLISHED_SPEEDUP = {32: 1.0, 64: 1.5, 128: 1.9, 256: 3.6, 512: 3.3}
PUBLISHED_FIG5_10X = {"1e-4": 65, "5e-5": 52, "1e-5": 20}      # percent of solves >=10x faster than fastest QDLDL
PUBLISHED_FIG6 = {("qdldl", 250): [21, 14, 8, 4, 2], ("qdldl", 500): [10, 6.5, 4, 2, 1],
                  ("qdldl", 1000): [4, 3, 1, None, None], ("pcg", 250): [22.2, 19.7, 15.4, 5.2, 4.4],
                  ("pcg", 500): [10.3, 10.6, 8, 4.6, 3], ("pcg", 1000): [4.9, 5.2, 3.7, 2.4, 1.7]}


def verdicts(run_dir: Path):
    out = {}
    for path in sorted(run_dir.glob("*/verdict.json")):
        verdict = json.loads(path.read_text())
        out.setdefault(verdict["workload"]["id"], []).append((verdict, path.parent))
    return out


def linsys(case: Path) -> np.ndarray:
    return np.loadtxt(case / "icra" / "trial_0_linsys_times.result", ndmin=1)


def figure4(v):
    print("| N | QDLDL (µs) | GBD-PCG (µs) | Speedup | Paper speedup | PCG mean L2 error (m) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    spread = {"pcg": 0.0, "qdldl": 0.0}
    for n in HORIZONS:
        means = {b: [linsys(c).mean() for _, c in v[f"icra-{b}-{n}-linsys-500hz"]] for b in ("qdldl", "pcg")}
        for b in means:
            spread[b] = max(spread[b], (max(means[b]) - min(means[b])) / np.mean(means[b]) * 100)
        q, p = np.mean(means["qdldl"]), np.mean(means["pcg"])
        err = np.mean([d["tracking_mean_l2"] for d, _ in v[f"icra-pcg-{n}-linsys-500hz"]])
        print(f"| {n} | {q:.0f} | {p:.0f} | {q / p:.1f}× | {PUBLISHED_SPEEDUP[n]:.1f}× | {err:.4f} |")
    print(f"\nRepeat spread (max-min over mean): PCG {spread['pcg']:.1f}%, QDLDL {spread['qdldl']:.1f}%")


def figure5(v):
    qd = np.concatenate([linsys(c) for _, c in v["icra-qdldl-128-linsys-500hz"]])
    fastest, p999 = qd.min(), np.percentile(qd, 99.9)
    print(f"\nQDLDL N=128: median {np.median(qd):.0f} µs, fastest {fastest:.0f} µs, 99.9th percentile {p999:.0f} µs, "
          f"slowest {qd.max():.0f} µs")
    print("\n| ε | Median (µs) | Mean (µs) | ≥10× faster than fastest QDLDL | Paper | Slowest / QDLDL p99.9 | ≥2× QDLDL p99.9 | Mean L2 error (m) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, key in [("10⁻⁴", "icra-pcg-128-linsys-500hz"), ("5·10⁻⁵", "icra-pcg-128-linsys-500hz-tol5e-5"),
                       ("10⁻⁵", "icra-pcg-128-linsys-500hz-tol1e-5")]:
        data = np.concatenate([linsys(c) for _, c in v[key]])
        err = np.mean([d["tracking_mean_l2"] for d, _ in v[key]])
        paper = PUBLISHED_FIG5_10X[{"10⁻⁴": "1e-4", "5·10⁻⁵": "5e-5", "10⁻⁵": "1e-5"}[label]]
        print(f"| {label} | {np.median(data):.0f} | {data.mean():.0f} | {np.mean(data < fastest / 10) * 100:.0f}% | {paper}% | "
              f"{data.max() / p999:.1f}× | {np.mean(data > 2 * p999) * 100:.1f}% | {err:.4f} |")


def figure6(v):
    print("\n| Solver | Rate | N = 32 | 64 | 128 | 256 | 512 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for backend, name in [("qdldl", "QDLDL"), ("pcg", "GBD-PCG")]:
        for rate in (250, 500, 1000):
            cells = []
            for j, n in enumerate(HORIZONS):
                runs = [d for d, _ in v[f"icra-{backend}-{n}-iters-{rate}hz"]]
                lost = max(d["tracking_max_l2"] for d in runs) > 1
                published = PUBLISHED_FIG6[(backend, rate)][j]
                shown = "lost" if lost else f"{np.mean([d['sqp_iters_mean'] for d in runs]):.1f}"
                cells.append(f"{shown} ({published if published is not None else 'X'})")
            print(f"| {name} | {rate if rate < 1000 else 1000} Hz | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    run = verdicts(Path(sys.argv[1]))
    figure4(run)
    figure5(run)
    figure6(run)
