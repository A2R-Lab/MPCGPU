#!/usr/bin/env python3
"""Render timing figures from MPCGPU timing-harness output (no measurement happens here).

Usage: plot_timing.py --icra DIR --fig8 DIR [--attribution DIR] --out DIR

--icra / --fig8 are `tools/timing.py run` output directories of the ICRA and figure-eight plans;
--attribution is a `tools/attribution/run.sh` results directory. Writes PNG files to --out.
Published values come from the ICRA 2024 paper text and Figure 6 table, never from bar heights.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, LINE, ACCENT, GREEN, GOLD = "#153453", "#52657a", "#dce2df", "#316045", "#8fbf6a", "#d9a441"
HORIZONS = [32, 64, 128, 256, 512]
CURRENT_ICRA = None
PUBLISHED_SPEEDUP = {32: 1.0, 64: 1.5, 128: 1.9, 256: 3.6, 512: 3.3}          # Figure 4 labels
PUBLISHED_FIG6 = {("qdldl", 250): [21, 14, 8, 4, 2], ("qdldl", 500): [10, 6.5, 4, 2, 1],
                  ("qdldl", 1000): [4, 3, 1, None, None], ("pcg", 250): [22.2, 19.7, 15.4, 5.2, 4.4],
                  ("pcg", 500): [10.3, 10.6, 8, 4.6, 3], ("pcg", 1000): [4.9, 5.2, 3.7, 2.4, 1.7]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def verdicts(run_dir: Path) -> dict[str, list[tuple[dict, Path]]]:
    out: dict[str, list] = {}
    for path in sorted(run_dir.glob("*/verdict.json")):
        verdict = json.loads(path.read_text())
        out.setdefault(verdict["workload"]["id"], []).append((verdict, path.parent))
    return out


def linsys(case: Path) -> np.ndarray:
    return np.loadtxt(case / "icra" / "trial_0_linsys_times.result", ndmin=1)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out / f"{name}.png")


def figure4(v, out):
    q = [np.mean([linsys(c).mean() for _, c in v[f"icra-qdldl-{n}-linsys-500hz"]]) for n in HORIZONS]
    p = [np.mean([linsys(c).mean() for _, c in v[f"icra-pcg-{n}-linsys-500hz"]]) for n in HORIZONS]
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, q, 0.38, color=GOLD, label="QDLDL (CPU)")
    ax.bar(x + 0.2, p, 0.38, color=INK, label="GBD-PCG (GPU)")
    for i, n in enumerate(HORIZONS):
        ax.annotate(f"{q[i] / p[i]:.1f}×", (x[i], max(q[i], p[i])), xytext=(0, 6), textcoords="offset points",
                    ha="center", color=ACCENT, fontweight="bold")
        ax.annotate(f"paper {PUBLISHED_SPEEDUP[n]:.1f}×", (x[i], max(q[i], p[i])), xytext=(0, 21),
                    textcoords="offset points", ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(x, [str(n) for n in HORIZONS])
    ax.set_xlabel("Trajectory length (knot points)")
    ax.set_ylabel("Average linear-system solve time (µs)")
    ax.set_ylim(0, max(q) * 1.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("ICRA pick-and-place, 500 Hz, current code on RTX 5090", color=INK, fontsize=11, loc="left")
    save(fig, out, "icra_linsys_time")


def figure5(v, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    qd = np.concatenate([linsys(c) for _, c in v["icra-qdldl-128-linsys-500hz"]])
    series = [("GBD-PCG ε = 1e-4", "icra-pcg-128-linsys-500hz", INK),
              ("GBD-PCG ε = 5e-5", "icra-pcg-128-linsys-500hz-tol5e-5", ACCENT),
              ("GBD-PCG ε = 1e-5", "icra-pcg-128-linsys-500hz-tol1e-5", GREEN)]
    for label, key, color in series:
        data = np.sort(np.concatenate([linsys(c) for _, c in v[key]]))
        ax.plot(data, np.linspace(0, 100, len(data)), color=color, label=label, lw=2)
    qd_sorted = np.sort(qd)
    ax.plot(qd_sorted, np.linspace(0, 100, len(qd_sorted)), color=GOLD, label="QDLDL (CPU)", lw=2)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Linear-system solve time (µs)")
    ax.set_ylabel("Percentage of solves")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Solve-time distribution, N = 128, three repeats", color=INK, fontsize=11, loc="left")
    save(fig, out, "icra_linsys_cdf")


def figure6(v, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    for ax, backend, label in [(axes[0], "qdldl", "MPCGPU with QDLDL"), (axes[1], "pcg", "MPCGPU with GBD-PCG")]:
        rates = [250, 500, 1000]
        values = np.zeros((3, 5))
        lost = np.zeros((3, 5), dtype=bool)
        for i, rate in enumerate(rates):
            for j, n in enumerate(HORIZONS):
                runs = [d for d, _ in v[f"icra-{backend}-{n}-iters-{rate}hz"]]
                values[i, j] = np.mean([d["sqp_iters_mean"] for d in runs])
                lost[i, j] = max(d["tracking_max_l2"] for d in runs) > 1
        ax.imshow(np.log1p(values), cmap="Greens", vmin=0, vmax=np.log1p(120), aspect="auto")
        for i, rate in enumerate(rates):
            for j in range(5):
                published = PUBLISHED_FIG6[(backend, rate)][j]
                text = "lost" if lost[i, j] else f"{values[i, j]:.1f}"
                ax.text(j, i - 0.12, text, ha="center", va="center", color=INK, fontweight="bold")
                ax.text(j, i + 0.25, f"paper {published if published is not None else '×'}", ha="center",
                        va="center", color=MUTED, fontsize=8)
        ax.set_xticks(range(5), [str(n) for n in HORIZONS])
        ax.set_yticks(range(3), ["250 Hz", "500 Hz", "1 kHz"])
        ax.set_xlabel("Knot points")
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Average SQP iterations per control step (current code; paper values below)", x=0.01,
                 ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, out, "icra_sqp_iterations")


def workspace(v, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(HORIZONS))
    styles = [("pcg", "reuse", INK, "GBD-PCG, reused workspace"), ("pcg", "fresh", "#8aa1b8", "GBD-PCG, rebuilt each solve"),
              ("qdldl", "reuse", GOLD, "QDLDL, reused workspace"), ("qdldl", "fresh", "#ecd29f", "QDLDL, rebuilt each solve")]
    for k, (backend, mode, color, label) in enumerate(styles):
        values = [statistics.median(d["median_us"] for d, _ in v[f"{backend}-{n}-{mode}"]) for n in HORIZONS]
        ax.bar(x - 0.3 + k * 0.2, values, 0.19, color=color, label=label)
    ax.set_xticks(x, [str(n) for n in HORIZONS])
    ax.set_xlabel("Trajectory length (knot points)")
    ax.set_ylabel("Median SQP time per control update (µs)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Figure-eight tracking: solver workspace reuse", color=INK, fontsize=11, loc="left")
    save(fig, out, "fig8_workspace")


def attribution(res: Path, out: Path) -> dict:
    """Per-iteration kernel slope (fixed 25 -> 100 iterations) and to-tolerance time per version."""
    rows = {}
    pattern = re.compile(r"repeat=(\d+) version=(\w+) BENCH N=(\d+) mode=(\w+) value=(\S+) iters=(\d+) capped=\d "
                         r"median_us=([\d.]+)")
    for line in (res / "kernel.txt").read_text().splitlines():
        m = pattern.search(line)
        if m:
            _, version, n, mode, value, iters, median = m.groups()
            rows.setdefault((version, int(n), mode, value), []).append((float(median), int(iters)))
    summary = {}
    for version in ["paper", "glass", "current"]:
        for n in HORIZONS:
            t25 = statistics.median(t for t, _ in rows[(version, n, "fixed", "25")])
            t100 = statistics.median(t for t, _ in rows[(version, n, "fixed", "100")])
            t1 = statistics.median(t for t, _ in rows[(version, n, "fixed", "1")])
            summary[(version, n)] = {"per_iter_us": (t100 - t25) / 75, "one_iter_us": t1}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(HORIZONS))
    for k, (version, color, label) in enumerate([("paper", GOLD, "Paper kernel (2024)"),
                                                ("glass", ACCENT, "Same kernel on GLASS"),
                                                ("current", INK, "Current kernel")]):
        ax.bar(x - 0.27 + k * 0.27, [summary[(version, n)]["per_iter_us"] for n in HORIZONS], 0.26,
               color=color, label=label)
    ax.set_xticks(x, [str(n) for n in HORIZONS])
    ax.set_xlabel("Trajectory length (knot points)")
    ax.set_ylabel("Time per PCG iteration (µs)")
    ax.legend(frameon=False)
    ax.set_title("GBD-PCG cost per iteration on identical systems (RTX 5090)", color=INK, fontsize=11, loc="left")
    save(fig, out, "attribution_per_iteration")
    paper_vs_current(res, out)
    return summary


PAPER_TOL = {32: "0.000005", 64: "0.000050", 128: "0.000100", 256: "0.000100", 512: "0.000100"}


def paper_vs_current(res: Path, out: Path, current: Path | None = None):
    """Mean GBD-PCG solve time and iterations: published code versus current code, both on this host."""
    current = current or CURRENT_ICRA
    paper_t, paper_i, cur_t, cur_i = [], [], [], []
    for n in HORIZONS:
        runs = sorted((res / "paper").glob("r*"))
        paper_t.append(np.median([np.loadtxt(r / f"tmp/results/{n}_PCG_{PAPER_TOL[n]}_0_linsys_times.result").mean() for r in runs]))
        paper_i.append(np.median([np.loadtxt(r / f"tmp/results/{n}_PCG_{PAPER_TOL[n]}_0_pcg_iters.result").mean() for r in runs]))
        cases = sorted(current.glob(f"icra-pcg-{n}-linsys-500hz-r*"))
        cur_t.append(np.mean([linsys(c).mean() for c in cases]))
        cur_i.append(np.mean([json.loads((c / "icra/summary.json").read_text())["linsys"]["mean_pcg_iters"] for c in cases]))
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, paper_t, 0.38, color=GOLD, label="Published code (2024)")
    ax.bar(x + 0.2, cur_t, 0.38, color=INK, label="Current code")
    for i in range(len(HORIZONS)):
        ax.annotate(f"{paper_i[i]:.0f} it", (x[i] - 0.2, paper_t[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=MUTED, fontsize=9)
        ax.annotate(f"{cur_i[i]:.0f} it", (x[i] + 0.2, cur_t[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(x, [str(n) for n in HORIZONS])
    ax.set_xlabel("Trajectory length (knot points)")
    ax.set_ylabel("Average GBD-PCG solve time (µs)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Same task and GPU: mean PCG iterations per solve labeled", color=INK, fontsize=11, loc="left")
    save(fig, out, "attribution_paper_vs_current")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icra", type=Path, required=True)
    parser.add_argument("--fig8", type=Path, required=True)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global CURRENT_ICRA
    CURRENT_ICRA = args.icra
    icra = verdicts(args.icra)
    figure4(icra, args.out)
    figure5(icra, args.out)
    figure6(icra, args.out)
    workspace(verdicts(args.fig8), args.out)
    if args.attribution:
        summary = attribution(args.attribution, args.out)
        for (version, n), row in sorted(summary.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"N={n:3d} {version:7s} per-iteration {row['per_iter_us']:.3f} us, one-iteration solve {row['one_iter_us']:.2f} us")


if __name__ == "__main__":
    main()
