#!/usr/bin/env python3
"""Generate MPCGPU's iiwa14 dynamics from its own URDF and pinned GRiD.

No sibling checkout, meshes, collision spherization, or runtime download is
needed. --check regenerates in a temporary directory and compares bytes.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = ROOT / "GRiD"
URDF = ROOT / "tools/iiwa14.urdf"
OUT = ROOT / "include/dynamics/iiwa/grid.cuh"


def generate(output: Path) -> None:
    # Freeze emission resource choices independently of shell tuning settings.
    for key, value in {"GRID_CUDA_TARGET_SHARED_MEM_BYTES": "98304",
                       "GRID_CUDA_TARGET_LITE_SHARED_MEM_BYTES": "49152",
                       "GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES": "4"}.items():
        os.environ[key] = value
    sys.path[:0] = [str(GRID_ROOT), str(GRID_ROOT / "external")]
    from URDFParser import URDFParser
    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

    robot = URDFParser().parse(str(URDF), floating_base=False)
    generator = GRiDCodeGenerator(robot, DEBUG_MODE=False,
                                  NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    generator.gen_all_code(
        include_homogenous_transforms=True,
        fixed_target_name="EE",
        codegen_profile="all",
        algorithm_list=["inverse_dynamics", "minv", "forward_dynamics",
                        "inverse_dynamics_gradient", "forward_dynamics_gradient",
                        "end_effector_pose", "end_effector_pose_gradient"],
        vendor_glass=False,
        output_path=str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="mpcgpu-codegen-") as tmp:
            candidate = Path(tmp) / "grid.cuh"
            generate(candidate)
            if candidate.read_bytes() != args.output.read_bytes():
                sys.exit("Generated header differs: run make regen and review the change")
        print("PASS reproducible iiwa14 code generation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate(args.output)
        print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
