#!/usr/bin/env python3
"""Regenerate MPCGPU's vendored GRiD CUDA header for iiwa14.

Writes:
    include/dynamics/iiwa/grid.cuh

Drives the GRiD code generator (vendored at the GRiD submodule) against the
vendored iiwa14 URDF — no robot_descriptions package needed. Run from the MPCGPU
repo root:

    python tools/regen_grid.py

Requires the GRiD submodule initialized:
    git submodule update --init --recursive GRiD

profile="all" is used so the generated header matches GATO's iiwa14 grid.cuh
byte-for-byte (same robot, same generator, same fixed EE target) — this is the
URDF-divergence gate, and it guarantees every `*_inner` the plant adapter calls
(forward_dynamics_inner / minv_inner / inverse_dynamics_inner[_vaf] /
inverse_dynamics_gradient_inner / end_effector_pose[_gradient]_inner /
load_update_X{I,matsHom}_helpers) plus grid_plant:: are present. MPCGPU does not
use GRiD's integrators, but carrying them (unused __device__ code) costs nothing
and keeps the header identical to GATO's for diffing.

Override the codegen location for an out-of-tree generator (e.g. reuse GATO's):
    GRID_ROOT=/path/to/GRiD python tools/regen_grid.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = Path(os.environ.get("GRID_ROOT", REPO_ROOT / "GRiD")).resolve()

URDF = REPO_ROOT / "tools" / "iiwa14.urdf"
OUT = REPO_ROOT / "include" / "dynamics" / "iiwa" / "grid.cuh"
FIXED_TARGET_NAME = "EE"   # iiwa14 fixed end-effector joint (matches GATO)

# GRiD packaging layout: grid_codegen at the GRiD root, URDFParser under
# external/ (both importable from a raw checkout; no pip install needed).
sys.path.insert(0, str(GRID_ROOT))
sys.path.insert(0, str(GRID_ROOT / "external"))


def main() -> None:
    if not (GRID_ROOT / "external" / "URDFParser").exists():
        sys.exit(f"GRiD not found at {GRID_ROOT}. Run: "
                 f"git submodule update --init --recursive GRiD "
                 f"(or set GRID_ROOT=/path/to/GRiD).")
    if not URDF.exists():
        sys.exit(f"URDF not found: {URDF}")

    from URDFParser import URDFParser
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[iiwa14] parsing {URDF}")
    robot = URDFParser().parse(str(URDF), floating_base=False)       # fixed base
    print(f"[iiwa14] EE target joint = '{FIXED_TARGET_NAME}'")

    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
                                FILE_NAMESPACE="grid")
    codegen.gen_all_code(
        include_homogenous_transforms=True,     # required for EE pose + gradient
        fixed_target_name=FIXED_TARGET_NAME,
        codegen_profile="all",
        output_path=str(OUT),
    )
    print(f"[iiwa14] wrote {OUT}")


if __name__ == "__main__":
    main()
