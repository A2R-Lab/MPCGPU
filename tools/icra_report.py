#!/usr/bin/env python3
"""Check and plot one ICRA 2024 pick-and-place run written by examples/icra_pick_place.cu.

Usage: icra_report.py RUN_DIR [--reference examples/icra/pick_place] [--urdf tools/iiwa14.urdf]

Writes RUN_DIR/report.json and RUN_DIR/trajectory.svg and exits nonzero if a check fails.
Checks: finite data, deterministic repeated trials, every reference offset reached in order, each of
the five goals visited, the final goal held, and the mean-error bound. The ICRA 2024 cost has no
joint-position, velocity or torque terms, so URDF limit excursions are reported, not enforced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOAL_RADIUS_M = 0.05      # closest approach to each goal during its segment
HOLD_RADIUS_M = 0.01      # distance to the final goal at the end of the circuit
MEAN_L2_BOUND_M = 0.10    # the paper reports ~10 cm average (L1) error for this task
SEGMENT_SPEED = 1.0       # rad/s: a segment starts where the reference joint speed rises past this


def urdf_limits(path: Path, joints: int) -> dict[str, np.ndarray]:
    rows = []
    for joint in ET.parse(path).getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") in {"revolute", "continuous"} and limit is not None:
            rows.append([float(limit.get(k)) for k in ("lower", "upper", "velocity", "effort")])
    rows = np.array(rows[:joints])
    return {"lower": rows[:, 0], "upper": rows[:, 1], "velocity": rows[:, 2], "effort": rows[:, 3]}


def segments(reference_qd: np.ndarray) -> list[tuple[int, int]]:
    speed = np.linalg.norm(reference_qd, axis=1)
    starts = [i for i in range(1, len(speed)) if speed[i] > SEGMENT_SPEED >= speed[i - 1]]
    ends = starts[1:] + [len(speed)]
    return list(zip(starts, ends))


def limit_summary(q, qd, u, limits) -> dict:
    below = np.maximum(limits["lower"] - q, 0).max(initial=0)
    above = np.maximum(q - limits["upper"], 0).max(initial=0)
    return {"position_excursion_rad": float(max(below, above)),
            "velocity_ratio_max": float((np.abs(qd) / limits["velocity"]).max()),
            "velocity_exceeded_fraction": float((np.abs(qd) > limits["velocity"]).any(axis=1).mean()),
            "torque_ratio_max": float((np.abs(u) / limits["effort"]).max())}


def svg(path: Path, ee, goal, reference_ee, goals, time, error):
    width, height, pad = 960, 360, 36
    panels = [("Top view (x, y)", 0, 1), ("Side view (x, z)", 0, 2)]
    points = np.vstack([ee, reference_ee])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height * 2}" '
             f'viewBox="0 0 {width} {height * 2}" font-family="sans-serif" font-size="13">',
             f'<rect width="{width}" height="{height * 2}" fill="#fff"/>']

    def polyline(xy, color, dash=""):
        text = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        return (f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="1.6"'
                f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')

    panel_width = width // 2
    for index, (title, a, b) in enumerate(panels):
        lo, hi = points[:, [a, b]].min(0), points[:, [a, b]].max(0)
        scale = min((panel_width - 2 * pad) / (hi[0] - lo[0]), (height - 2 * pad) / (hi[1] - lo[1]))
        def xy(data):
            x = index * panel_width + pad + (data[:, a] - lo[0]) * scale
            return np.stack([x, height - pad - (data[:, b] - lo[1]) * scale], 1)
        parts.append(f'<text x="{index * panel_width + pad}" y="22">{title}</text>')
        parts.append(polyline(xy(reference_ee), "#9aa4b2", "5 4"))
        parts.append(polyline(xy(ee), "#1f5fbf"))
        for number, point in enumerate(xy(goals), 1):
            parts.append(f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="5" fill="#d1495b"/>'
                         f'<text x="{point[0] + 7:.1f}" y="{point[1] - 7:.1f}">{number}</text>')
    top, span = height + pad, height - 2 * pad
    scale_y = span / max(error.max(), 1e-9)
    scale_x = (width - 2 * pad) / time[-1]
    trace = np.stack([pad + time * scale_x, top + span - error * scale_y], 1)
    parts.append(f'<text x="{pad}" y="{height + 22}">L2 end-effector error (m) over time (s); '
                 f'max {error.max():.3f} m</text>')
    parts.append(f'<line x1="{pad}" y1="{top + span}" x2="{width - pad}" y2="{top + span}" stroke="#555"/>')
    parts.append(polyline(trace, "#1f5fbf"))
    for second in range(0, int(time[-1]) + 1, 2):
        x = pad + second * scale_x
        parts.append(f'<line x1="{x:.1f}" y1="{top + span}" x2="{x:.1f}" y2="{top + span + 5}" stroke="#555"/>'
                     f'<text x="{x:.1f}" y="{top + span + 20}" text-anchor="middle">{second}</text>')
    parts.append(f'<text x="{width - pad}" y="{height + 22}" text-anchor="end" fill="#555">solid: simulated robot · '
                 f'dashed: reference · red: goals</text></svg>')
    path.write_text("\n".join(parts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", default=str(ROOT / "examples/icra/pick_place"))
    parser.add_argument("--urdf", type=Path, default=ROOT / "tools/iiwa14.urdf")
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text())
    samples = np.genfromtxt(args.run_dir / "samples.csv", delimiter=",", names=True)
    updates = np.genfromtxt(args.run_dir / "updates.csv", delimiter=",", names=True)
    reference_xu = np.loadtxt(args.reference + "_traj.csv", delimiter=",", ndmin=2)
    reference_ee = np.loadtxt(args.reference + "_eepos.traj", delimiter=",", ndmin=2)[:, :3]
    joints = reference_xu.shape[1] // 3
    limits = urdf_limits(args.urdf, joints)

    ee = np.stack([samples[f"ee_{a}"] for a in "xyz"], 1)
    goal = np.stack([samples[f"goal_{a}"] for a in "xyz"], 1)
    q = np.stack([updates[f"q{i}"] for i in range(joints)], 1)
    qd = np.stack([updates[f"qd{i}"] for i in range(joints)], 1)
    u = np.stack([updates[f"u{i}"] for i in range(joints)], 1)
    error = np.linalg.norm(ee - goal, axis=1)
    time = samples["time_s"]

    goal_rows = []
    for start, end in segments(reference_xu[:, joints:2 * joints]):
        target = reference_ee[end - 1]
        distance = np.linalg.norm(ee[start:end] - target, axis=1)
        closest = int(distance.argmin())
        goal_rows.append({"start_row": start, "end_row": end, "goal_m": target.round(6).tolist(),
                          "closest_approach_m": float(distance[closest]),
                          "closest_time_s": float((start + closest) * summary["config"]["timestep_s"]),
                          "error_at_end_m": float(distance[-1])})

    run_limits = limit_summary(q, qd, u, limits)
    reference_limits = limit_summary(reference_xu[:, :joints], reference_xu[:, joints:2 * joints],
                                     reference_xu[:, 2 * joints:], limits)
    checks = {
        "finite": bool(np.isfinite(ee).all() and np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(u).all()),
        "deterministic": bool(summary["deterministic"]),
        "all_offsets": len(ee) == len(reference_ee) == summary["reference_rows"],
        "goal_sequence": bool(np.allclose(goal, reference_ee, atol=1e-6)),
        "five_goals": len(goal_rows) == 5,
        "goals_visited": all(row["closest_approach_m"] < GOAL_RADIUS_M for row in goal_rows),
        "final_goal_held": bool(goal_rows) and goal_rows[-1]["error_at_end_m"] < HOLD_RADIUS_M,
        "mean_error": float(error.mean()) < MEAN_L2_BOUND_M,
    }
    report = {"run": str(args.run_dir), "config": summary["config"], "checks": checks,
              "passed": all(checks.values()),
              "error_m": {"l2_mean": float(error.mean()), "l2_max": float(error.max()), "l2_final": float(error[-1]),
                          "l1_mean": float(np.abs(ee - goal).sum(1).mean())},
              "goals": goal_rows, "limits": {"run": run_limits, "reference": reference_limits},
              "solver": {"sqp": summary["sqp"], "linsys": summary["linsys"]},
              "thresholds": {"goal_radius_m": GOAL_RADIUS_M, "hold_radius_m": HOLD_RADIUS_M,
                             "mean_l2_bound_m": MEAN_L2_BOUND_M}}
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    svg(args.run_dir / "trajectory.svg", ee, goal, reference_ee, np.array([r["goal_m"] for r in goal_rows]), time, error)
    print(f"{'PASS' if report['passed'] else 'FAIL'} {args.run_dir}: mean L2 {error.mean():.4f} m, "
          f"mean L1 {report['error_m']['l1_mean']:.4f} m, closest approach "
          + " ".join(f"{row['closest_approach_m']:.4f}" for row in goal_rows)
          + f" m, joint excursion {run_limits['position_excursion_rad']:.3f} rad")
    for name, ok in checks.items():
        if not ok:
            print(f"  failed check: {name}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
