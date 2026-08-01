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

# Prefer the sibling GATO checkout's URDF: it is byte-identical to the vendored
# tools/iiwa14.urdf BUT sits next to the link STL meshes, which the collision
# spherization (collision_spec_from_urdf) resolves relative to the URDF file.
# The lone vendored copy silently degrades the sphere set (meshes unresolvable)
# and breaks the grid.cuh byte-diff gate vs GATO. Override with GRID_URDF=.
_GATO_URDF = Path(os.environ.get("GATO_ROOT", REPO_ROOT.parent / "GATO")) \
    / "examples" / "iiwa_description" / "iiwa14.urdf"
URDF = Path(os.environ["GRID_URDF"]) if "GRID_URDF" in os.environ else (
    _GATO_URDF if _GATO_URDF.exists() else REPO_ROOT / "tools" / "iiwa14.urdf")
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
    # collision_spec + contact_frames mirror GATO's builder.codegen defaults
    # (collision_res=0.15, contact_frames=[ee_frame]) — GATO's iiwa14 grid.cuh
    # bakes the grid_collision namespace + EE wrench map since its CL-2b regen,
    # and the byte-diff gate (test/test_gates.py::test_grid_cuh_matches_gato)
    # requires identical codegen inputs. MPCGPU does not call either namespace;
    # carrying them (unused __device__ code) costs nothing.
    from grid_codegen.algorithms._collision import collision_spec_from_urdf
    from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
    codegen.gen_all_code(
        include_homogenous_transforms=True,     # required for EE pose + gradient
        fixed_target_name=FIXED_TARGET_NAME,
        codegen_profile="all",
        output_path=str(OUT),
        collision_spec=collision_spec_from_urdf(robot, str(URDF), resolution=0.15),
        contact_frames=contact_frames_from_urdf(robot, [FIXED_TARGET_NAME]),
    )
    print(f"[iiwa14] wrote {OUT}")


if __name__ == "__main__":
    main()
