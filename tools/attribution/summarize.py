#!/usr/bin/env python3
"""Summarize a tools/attribution/run.sh results directory against the current ICRA timing run.

Usage: summarize.py RESULTS_DIR CURRENT_ICRA_TIMING_DIR
Prints Markdown tables: per-iteration kernel cost by version, and paper-era versus current
linear-system times on the RTX 5090 next to the published RTX 4090 speedups.
"""
import collections
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

HORIZONS = [32, 64, 128, 256, 512]
PUBLISHED_SPEEDUP = {32: 1.0, 64: 1.5, 128: 1.9, 256: 3.6, 512: 3.3}
# Middle entry of the 2024 per-horizon tolerance sweeps, as the current task build uses.
PAPER_TOL = {32: "0.000005", 64: "0.000050", 128: "0.000100", 256: "0.000100", 512: "0.000100"}


def kernel_table(res: Path):
    rows = collections.defaultdict(list)
    pattern = re.compile(r"version=(\w+) BENCH N=(\d+) mode=(\w+) value=(\S+) iters=\d+ capped=\d median_us=([\d.]+)")
    for line in (res / "kernel.txt").read_text().splitlines():
        m = pattern.search(line)
        if m:
            version, n, mode, value, median = m.groups()
            rows[(version, int(n), mode, value)].append(float(median))
    med = lambda key: statistics.median(rows[key])
    print("| N | Paper kernel (µs/iter) | On GLASS (µs/iter) | Current (µs/iter) | GLASS change | One-iteration solve, paper / current (µs) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in HORIZONS:
        per = {v: (med((v, n, "fixed", "100")) - med((v, n, "fixed", "25"))) / 75 for v in ("paper", "glass", "current")}
        print(f"| {n} | {per['paper']:.2f} | {per['glass']:.2f} | {per['current']:.2f} | "
              f"{(per['glass'] / per['paper'] - 1) * 100:+.1f}% | {med(('paper', n, 'fixed', '1')):.1f} / "
              f"{med(('current', n, 'fixed', '1')):.1f} |")


def load(path: Path):
    return np.loadtxt(path, ndmin=1) if path.exists() and path.stat().st_size else np.array([])


def paper_table(res: Path, current: Path):
    print("\n| N | Paper code PCG (µs, median [range]) | Paper code iterations | Current PCG (µs) | Current iterations | "
          "Paper code QDLDL (µs) | Current QDLDL (µs) | Paper code speedup | Current speedup | Published speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in HORIZONS:
        q, p, it = [], [], []
        for rdir in sorted((res / "paper").glob("r*")):
            base = rdir / "tmp/results"
            q.append(load(base / f"{n}_QDLDL_0_linsys_times.result").mean())
            p.append(load(base / f"{n}_PCG_{PAPER_TOL[n]}_0_linsys_times.result").mean())
            it.append(load(base / f"{n}_PCG_{PAPER_TOL[n]}_0_pcg_iters.result").mean())
        cases = sorted(current.glob(f"icra-pcg-{n}-linsys-500hz-r*"))
        cp = np.median([load(c / "icra/trial_0_linsys_times.result").mean() for c in cases])
        ci = np.median([json.loads((c / "icra/summary.json").read_text())["linsys"]["mean_pcg_iters"] for c in cases])
        cq = np.median([load(c / "icra/trial_0_linsys_times.result").mean()
                        for c in current.glob(f"icra-qdldl-{n}-linsys-500hz-r*")])
        pm, qm = np.median(p), np.median(q)
        print(f"| {n} | {pm:.0f} [{min(p):.0f}–{max(p):.0f}] | {np.median(it):.1f} | {cp:.0f} | {ci:.1f} | {qm:.0f} | {cq:.0f} | "
              f"{qm / pm:.1f}× | {cq / cp:.1f}× | {PUBLISHED_SPEEDUP[n]:.1f}× |")


if __name__ == "__main__":
    results, current = Path(sys.argv[1]), Path(sys.argv[2])
    kernel_table(results)
    paper_table(results, current)
