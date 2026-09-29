#!/usr/bin/env python3
"""Sequential, content/configuration-aware CUDA builds. Never runs a benchmark."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "pcg": "tools/validate_track.cu",
    "qdldl": "tools/validate_track.cu",
    "single-cost": "tools/single_cost_test.cu",
    "terminal": "tools/test_terminal_cost.cu",
    "fd": "examples/test_fd_parity.cu",
    "reference": "tools/gen_reference.cu",
    "ee-from-joints": "tools/ee_from_joints.cu",
    "icra-pcg": "examples/icra_pick_place.cu",
    "icra-qdldl": "examples/icra_pick_place.cu",
    "gbd-spd": "GBD-PCG/examples/test_pcg_spd.cu",
    "gbd-api": "GBD-PCG/examples/test_api.cu",
    "gbd-matvec": "GBD-PCG/examples/test_bdmv.cu",
    "workspace": "test/workspace.cu",
    "model-oracle": "test/model_oracle.cu",
    "gbd-dumped": "GBD-PCG/examples/test_pcg_dumped.cu",
}


# Figure-eight tracking configuration used by the maintained demos and gates.
FIG8_FLAGS = ["-DPCG_MAX_ITER=200", "-DPCG_RES_TOL=1e-4", "-DGATO_REG_PATTERN",
              "-DRHO_INIT=0.01", "-DSQP_MAX_ITER=1", "-DSQP_MAX_TIME_US=100000000"]


def icra_flags(knots: int) -> list[str]:
    """ICRA 2024 pick-and-place protocol; see docs/icra-replication.md for sources and changes."""
    return ["-DTIMESTEP=0.015625", "-DEE_COST=1", "-DN_COST=1", "-DQ_COST=0", "-DQD_COST=1e-4",
            f"-DU_COST={1e-3 if knots == 64 else 1e-4}", "-DQ_LIM_COST=0", "-DVEL_LIM_COST=0",
            "-DCTRL_LIM_COST=0", "-DRHO_INIT=1e-3", "-DPCG_RES_TOL=0", "-DSQP_MAX_ITER=20",
            "-DWARMUP_RESET=1", "-DREFERENCE_TAIL_FILL=1", "-DMPCGPU_GRAVITY=0"]


def source_digest() -> str:
    digest = hashlib.sha256()
    for directory in ("include", "GBD-PCG/include", "GLASS", "qdldl/include"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix in {".h", ".hpp", ".cuh"} and path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    for submodule in ("GLASS", "GRiD", "qdldl"):
        digest.update(subprocess.check_output(["git", "-C", str(ROOT / submodule), "rev-parse", "HEAD"]))
    library = ROOT / "qdldl/build/out/libqdldl.so"
    if library.exists():
        digest.update(library.read_bytes())
    return digest.hexdigest()


def build(target: str, *, knots=64, states=14, dtype="float", arch=None,
          profile="correctness", output=None, extra="", nvcc="nvcc") -> Path:
    if target not in SOURCES or knots < 1 or states < 1 or dtype not in {"float", "double"} or profile not in {"correctness", "timing"}:
        raise ValueError("Unknown target or nonpositive dimensions")
    arch = arch or os.environ.get("ARCH", "sm_120")
    output = Path(output) if output else ROOT / "build" / profile / f"{target}-{states}x{knots}-{dtype}.exe"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", f"-arch={arch}", "-Iinclude", "-Iinclude/common",
             "-IGLASS", "-IGBD-PCG/include", "-Iqdldl/include",
             f"-DSTATE_SIZE={states}", f"-DKNOT_POINTS={knots}"]
    if target.startswith("gbd-"):
        flags += [f"-DTEST_DOUBLE={int(dtype == 'double')}"]
    else:
        if states != 14 or knots < 2:
            raise ValueError("MPCGPU requires 14 iiwa states and a horizon >=2")
        if dtype != "float":
            raise ValueError("MPCGPU's QDLDL ABI is float; double is supported by GBD-PCG only")
        flags += icra_flags(knots) if target.startswith("icra-") else FIG8_FLAGS
        flags += ["-Lqdldl/build/out", "-lqdldl", "-lcublas"]
        if target in {"qdldl", "icra-qdldl"}:
            flags += ["-DLINSYS_SOLVE=0"]
    flags += (["-DMPCGPU_CORRECTNESS=1", "-DTIME_LINSYS=0"] if profile == "correctness"
              else ["-DSAVE_DATA=1"])
    command = [nvcc, *flags, *shlex.split(extra), SOURCES[target], "-o", str(output)]
    (ROOT / "build").mkdir(exist_ok=True)
    with (ROOT / "build/.compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiler = subprocess.check_output([nvcc, "--version"], text=True)
        recipe = {"command": command, "compiler": compiler, "headers": source_digest(),
                  "source": hashlib.sha256((ROOT / SOURCES[target]).read_bytes()).hexdigest(),
                  "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        stamp = output.with_suffix(".build.json")
        if output.exists() and stamp.exists():
            saved = json.loads(stamp.read_text())
            if saved.get("recipe") == recipe and saved.get("binary_sha256") == hashlib.sha256(output.read_bytes()).hexdigest():
                return output
        subprocess.run(command, cwd=ROOT, check=True)
        stamp.write_text(json.dumps({"recipe": recipe,
            "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=SOURCES)
    parser.add_argument("--knots", type=int, default=64)
    parser.add_argument("--states", type=int, default=14)
    parser.add_argument("--dtype", choices=["float", "double"], default="float")
    parser.add_argument("--profile", choices=["correctness", "timing"], default="correctness")
    parser.add_argument("--arch")
    parser.add_argument("--output")
    parser.add_argument("--extra", default="")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    print(build(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
