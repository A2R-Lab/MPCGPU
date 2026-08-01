"""MPCGPU correctness gates as a pytest suite (the gpu-proof receipt surface).

Thin wrapper over the existing gate runners — the gates themselves stay
shell-first (tools/run_gates.sh is still the human entry point); this module
runs each runner ONCE per session and asserts the per-gate verdicts, so the
signed receipt (test/run_gpu_proof.sh) records one outcome per gate:

  - tools/run_gates.sh          -> single_cost_test, test_terminal_cost,
                                   validate_track_pcg, validate_track_qdldl
  - make test_fd_parity         -> adapter forwardDynamics vs grid:: reference
  - grid.cuh byte-diff vs GATO  -> URDF-divergence gate (GATO_ROOT env, or the
                                   sibling checkout ../GATO)
  - GBD-PCG/test/run_gates.sh   -> cooperative bdmv/PCG SPD gates

Prereqs (test/run_gpu_proof.sh handles these): GPU box, qdldl built
(make build_qdldl), submodules checked out.
"""
import os
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "LD_LIBRARY_PATH": str(ROOT / "qdldl/build/out")}


@pytest.fixture(scope="session")
def gates():
    """One tools/run_gates.sh pass; individual tests assert per-gate lines."""
    r = subprocess.run(["bash", "tools/run_gates.sh"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=3600)
    return r


def _passed(gates, name):
    # runner lines: "PASS  <name>  <detail>" / "FAIL  <name>  <detail>"
    for line in gates.stdout.splitlines():
        m = re.match(r"^(PASS|FAIL)\s+(\S+)", line)
        if m and m.group(2) == name:
            return m.group(1) == "PASS", line
    return False, f"gate line for {name!r} not found\n--- stdout ---\n{gates.stdout}\n--- stderr ---\n{gates.stderr}"


@pytest.mark.parametrize("gate_name", [
    "single_cost_test",
    "test_terminal_cost",
    "validate_track_pcg",
    "validate_track_qdldl",
])
def test_gate(gates, gate_name):
    ok, detail = _passed(gates, gate_name)
    assert ok, detail


def test_gate_runner_exit(gates):
    assert gates.returncode == 0, f"tools/run_gates.sh rc={gates.returncode}\n{gates.stdout}\n{gates.stderr}"


def test_fd_parity():
    """Adapter forwardDynamics vs grid::forward_dynamics_device (expect ~0)."""
    b = subprocess.run(["make", "test_fd_parity"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=1200)
    assert b.returncode == 0, f"build failed:\n{b.stdout}\n{b.stderr}"
    r = subprocess.run(["./examples/test_fd_parity.exe"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "PASS" in r.stdout, f"{r.stdout}\n{r.stderr}"


def test_grid_cuh_matches_gato():
    """URDF-divergence gate: the generated iiwa14 grid.cuh must be byte-identical
    to GATO's (both regenerate from the same pinned GRiD)."""
    gato_root = pathlib.Path(os.environ.get("GATO_ROOT", ROOT.parent / "GATO"))
    theirs = gato_root / "gato/dynamics/iiwa14/grid.cuh"
    ours = ROOT / "include/dynamics/iiwa/grid.cuh"
    assert theirs.is_file(), f"GATO checkout not found at {gato_root} (set GATO_ROOT)"
    assert ours.read_bytes() == theirs.read_bytes(), \
        "grid.cuh diverged from GATO's iiwa14 grid.cuh — regenerate both from the pinned GRiD"


def test_gbdpcg_gates():
    """GBD-PCG's own gate runner (cooperative bdmv + PCG SPD residual gates).
    PYTHON = this interpreter (the runner's synthetic-strip gate needs numpy)."""
    import sys
    r = subprocess.run(["bash", "test/run_gates.sh"], cwd=ROOT / "GBD-PCG",
                       env={**ENV, "PYTHON": sys.executable},
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"GBD-PCG gates rc={r.returncode}\n{r.stdout}\n{r.stderr}"
