"""Individually attested, deterministic GPU gates; never read ambient /tmp data."""
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build import build

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_proof]
ENV = {**os.environ, "LD_LIBRARY_PATH": str(ROOT / "qdldl/build/out") + ":" + os.environ.get("LD_LIBRARY_PATH", "")}


def run(executable, *args):
    result = subprocess.run([str(executable), *map(str, args)], cwd=ROOT, env=ENV,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="session")
def binary():
    cache = {}
    def get(target, **kwargs):
        key = (target, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = build(target, **kwargs)
        return cache[key]
    return get


def test_single_cost(binary):
    output = run(binary("single-cost"), "examples/trajfiles/0_0")
    step = float(re.search(r"d_xu_before \|\| = ([\d.eE+-]+)", output)[1])
    post = float(re.search(r"\[post\] mean \|EE_k - goal_k\| over window = ([\d.eE+-]+)", output)[1])
    assert 300 <= step <= 380 and 0.005 <= post <= 0.02, output


def test_terminal_cost(binary):
    output = run(binary("terminal"))
    rows = {}
    for line in output.splitlines():
        if "q-block:" in line:
            rows[line[1]] = np.fromstring(line.split("q-block:")[1], sep=" ")
        elif "pin truth" in line:
            rows["truth"] = np.fromstring(line.split(":")[1], sep=" ")
    assert set(rows) == {"A", "B", "C", "D", "truth"}, output
    for row in rows.values():
        assert row.shape == (7,) and np.isfinite(row).all(), output
        np.testing.assert_allclose(row, rows["A"], atol=0.05, rtol=0)


@pytest.mark.parametrize("backend", ["pcg", "qdldl"])
def test_tracking(binary, backend, tmp_path):
    output = run(binary(backend), "examples/trajfiles/0_0", tmp_path / "tracking")
    row = re.search(r"RESULT offsets=(\d+) mean=([\d.eE+-]+) max=([\d.eE+-]+) final=([\d.eE+-]+)", output)
    assert row, output
    assert int(row[1]) == 1202, output
    values = np.array([float(v) for v in row.groups()[1:]])
    assert np.isfinite(values).all() and 0.02 <= values[0] <= 0.05 and values[1] < 0.08, output


def test_fd_parity(binary):
    assert "PASS" in run(binary("fd"))


@pytest.mark.parametrize("dims", [(2,3), (6,8), (14,32)])
def test_gbd_spd(binary, dims):
    assert "PASS" in run(binary("gbd-spd", states=dims[0], knots=dims[1]))


@pytest.mark.parametrize("dims,dtype", [((1,1),"float"), ((2,3),"float"), ((6,7),"float"), ((14,32),"float"), ((2,3),"double")])
@pytest.mark.parametrize("case", ["host", "device", "zero", "zero-exact", "warm", "cap", "invalid"])
def test_gbd_api(binary, dims, dtype, case):
    assert "PASS" in run(binary("gbd-api", states=dims[0], knots=dims[1], dtype=dtype), case)


def test_gbd_matvec(binary, tmp_path):
    path = tmp_path / "strips.bin"
    np.random.default_rng(0).uniform(-1, 1, 3*14*14*32).astype(np.float32).tofile(path)
    output = run(binary("gbd-matvec", knots=32), path)
    error = float(re.search(r"\(rel ([\d.eE+-]+)\)", output)[1])
    assert np.isfinite(error) and error < 1e-4, output


@pytest.mark.parametrize("backend", ["pcg", "qdldl"])
def test_workspace_tracking_parity(binary, backend, tmp_path):
    reused = run(binary(backend), "examples/trajfiles/0_0", tmp_path/"reused")
    fresh = run(binary(backend, extra="-DMPCGPU_NO_REUSE=1", output=str(tmp_path/"fresh.exe")),
                "examples/trajfiles/0_0", tmp_path/"fresh")
    def state_hash(output):
        match = re.search(r"STATE_HASH\s+(\S+)", output)
        assert match, output
        return match[1]
    assert state_hash(reused) == state_hash(fresh)


def test_workspace_contract(binary):
    assert "PASS" in run(binary("workspace"))


def test_independent_model_oracle(binary):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(ROOT/"tools/iiwa14.urdf"))
    data = model.createData()
    ee = model.getFrameId("EE")
    rows = [np.fromstring(line[7:], sep=" ") for line in run(binary("model-oracle")).splitlines()
            if line.startswith("ORACLE ")]
    assert len(rows) == 5
    for row in rows:
        assert row.shape == (230,) and np.isfinite(row).all()
        q,v,u,a = row[:28].reshape(4,7)
        derivatives = row[28:175].reshape(7,21,order="F")
        pose,jac,bias = row[175:181],row[181:223].reshape(6,7,order="F"),row[223:]
        np.testing.assert_allclose(a,pin.aba(model,data,q,v,u),rtol=3e-4,atol=3e-4)
        dq,dv,du = pin.computeABADerivatives(model,data,q,v,u)
        np.testing.assert_allclose(derivatives,np.hstack((dq,dv,du)),rtol=5e-4,atol=2e-3)
        np.testing.assert_allclose(bias,pin.rnea(model,data,q,v,np.zeros(7)),rtol=3e-4,atol=3e-4)
        pin.framesForwardKinematics(model,data,q)
        np.testing.assert_allclose(pose[:3],data.oMf[ee].translation,rtol=0,atol=2e-6)
        expected_jac=pin.computeFrameJacobian(model,data,q,ee,pin.LOCAL_WORLD_ALIGNED)
        np.testing.assert_allclose(jac[:3],expected_jac[:3],rtol=0,atol=3e-6)


@pytest.mark.parametrize("extra", ["-DPCG_TRUE_EXIT_CHECK_PERIOD=2", "-DPCG_RESIDUAL_REPLACE_PERIOD=2"])
def test_gbd_residual_modes(binary, tmp_path, extra):
    for case in ("device", "warm", "zero-exact"):
        assert "PASS" in run(binary("gbd-api",states=2,knots=3,extra=extra,
                                    output=str(tmp_path/"variant.exe")),case)


def test_real_schur_fixture(binary, tmp_path, monkeypatch):
    monkeypatch.setitem(ENV,"MPCGPU_DUMP_DIR",str(tmp_path))
    executable=binary("pcg",knots=32,extra="-DDUMP_KKT -DDUMP_KKT_AT_SOLVE=100",
                      output=str(tmp_path/"capture.exe"))
    run(executable,"examples/trajfiles/0_0",tmp_path/"capture")
    for name,count in (("S",3*14*14*32),("Pinv",3*14*14*32),("gamma",14*32)):
        values=np.fromfile(tmp_path/f"mpc_{name}.bin",dtype=np.float32)
        assert values.size==count and np.isfinite(values).all()
    assert "PASS" in run(binary("gbd-dumped",knots=32,extra="-DPCG_TRUE_EXIT_CHECK_PERIOD=1"),tmp_path)


def test_icra_reference_targets(binary, tmp_path):
    # The tracked ICRA targets must equal FK of the verbatim joint reference under the current model.
    regenerated = tmp_path / "pick_place_eepos.traj"
    run(binary("ee-from-joints"), "examples/icra/pick_place_traj.csv", regenerated)
    np.testing.assert_allclose(np.loadtxt(regenerated, delimiter=","),
                               np.loadtxt(ROOT / "examples/icra/pick_place_eepos.traj", delimiter=","),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend", ["pcg", "qdldl"])
def test_icra_pick_place(binary, backend, tmp_path):
    # Deterministic ICRA 2024 circuit at N=64: two identical trials and every task-quality check.
    out = tmp_path / "icra"
    run(binary(f"icra-{backend}"), "--out", out, "--trials", 2)
    result = subprocess.run([sys.executable, "tools/icra_report.py", str(out)], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
