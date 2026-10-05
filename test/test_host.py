"""CUDA-free contract gates, also run in ordinary CI."""
import hashlib
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu_proof


def test_codegen_reproducible():
    result = subprocess.run([sys.executable, "tools/regen_grid.py", "--check"], cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dependency_pins():
    for name, expected in json.loads((ROOT / "tools/dependencies.json").read_text()).items():
        actual = subprocess.check_output(["git", "-C", str(ROOT/name), "rev-parse", "HEAD"], text=True).strip()
        assert actual == expected, f"{name}: update the reviewed dependency manifest with the pin"


def test_gbd_pcg_glass_pin_lockstep():
    # MPCGPU compiles GBD-PCG against its own top-level GLASS; the submodule's nested GLASS pin
    # must be the same commit so the standalone GBD-PCG build sees what the receipt attested.
    nested = subprocess.check_output(["git", "-C", str(ROOT / "GBD-PCG"), "ls-tree", "HEAD", "GLASS"], text=True).split()[2]
    top = subprocess.check_output(["git", "-C", str(ROOT / "GLASS"), "rev-parse", "HEAD"], text=True).strip()
    assert nested == top, "bump GBD-PCG's GLASS submodule and MPCGPU's GLASS pin together"


def test_receipt_policy():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["gpu_proof"]
    policy = yaml.safe_load((ROOT / "test/gpu-proof-policy.yaml").read_text())
    assert set(config["fingerprint_paths"]) == set(policy["required_fingerprint_paths"])
    assert config["fingerprint_excluded_paths"] == policy["required_fingerprint_excluded_paths"]
    assert policy["min_schema"] == 3 and policy["allow_dirty"] is False and policy["allow_carried"] is False
    assert "pytest-gpu-proof==0.4.0" in (ROOT / "requirements-dev.txt").read_text()


@pytest.fixture(scope="session")
def parser_binary(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("csv-parser")
    source = tmp / "parser.cpp"
    source.write_text('''#include "utils/trajectory.hpp"
#include <iostream>
int main(int argc, char** argv) {
    try { mpcgpu::Reference<float> r(argv[1], 2, 14, 7); }
    catch (const std::exception& e) { std::cerr << e.what(); return 1; }
    return 0;
}
''')
    executable = tmp / "parser"
    subprocess.run(["c++", "-std=c++17", "-I", str(ROOT/"include"), str(source), "-o", str(executable)], check=True)
    return executable


@pytest.mark.parametrize("case", ["valid", "missing", "empty", "ragged", "nan", "inf", "short", "mismatch", "trailing", "garbage"])
def test_trajectory_validation(parser_binary, tmp_path, case):
    ee = [",".join(["0"]*6)]*3
    xu = [",".join(["0"]*21)]*3
    if case == "empty": ee = []
    if case == "ragged": xu[1] = "1,2"
    if case in {"nan", "inf", "garbage"}: xu[1] = xu[1].replace("0", {"nan":"nan","inf":"inf","garbage":"1junk"}[case], 1)
    if case == "short": ee, xu = ee[:1], xu[:1]
    if case == "mismatch": xu = xu[:2]
    if case == "trailing": ee[1] += ","
    prefix = tmp_path / "reference"
    if case != "missing": Path(str(prefix)+"_eepos.traj").write_text("\n".join(ee))
    Path(str(prefix)+"_traj.csv").write_text("\n".join(xu))
    result = subprocess.run([str(parser_binary), str(prefix)], capture_output=True)
    assert (result.returncode == 0) == (case == "valid")


def test_builder_configuration_cache(tmp_path, monkeypatch):
    # A single -D flag must remain the value of --extra, not a new CLI option.
    commands=subprocess.check_output(['make','--dry-run','examples','EXTRA_FLAGS=-DUNIT_TEST'],cwd=ROOT,text=True)
    build_commands=[command for command in commands.splitlines() if 'tools/build.py' in command]
    assert len(build_commands) == 2
    assert all('--extra=-DUNIT_TEST' in shlex.split(command) for command in build_commands)
    spec = importlib.util.spec_from_file_location("test_builder", ROOT / "tools/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/validate_track.cu").write_text("source")
    monkeypatch.setattr(builder, "source_digest", lambda: "headers-v1")
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: "compiler-v1")
    calls = []
    def compile(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"compiled-binary")
    monkeypatch.setattr(builder.subprocess, "run", compile)
    output = tmp_path / "demo.exe"
    builder.build("pcg", output=output, arch="sm_120")
    builder.build("pcg", output=output, arch="sm_120")
    assert len(calls) == 1
    builder.build("pcg", output=output, arch="sm_86")
    assert len(calls) == 2
    builder.build("pcg", output=output, arch="sm_86", knots=32)
    assert len(calls) == 3
    output.write_bytes(b"stale-or-replaced")
    builder.build("pcg", output=output, arch="sm_86", knots=32)
    assert len(calls) == 4
    monkeypatch.setattr(builder, "source_digest", lambda: "headers-v2")
    builder.build("pcg", output=output, arch="sm_86", knots=32)
    assert len(calls) == 5


def test_receipt_refuses_partial_collection():
    for args, addopts in ((["-k", "one_test"], ""), ([], "-k one_test")):
        result = subprocess.run(["bash", "test/run_gpu_proof.sh", *args], cwd=ROOT,
                                env={**os.environ, "PYTEST_ADDOPTS": addopts},
                                capture_output=True, text=True)
        assert result.returncode != 0 and "full suite" in result.stderr


def test_expected_collection_manifest():
    output=subprocess.check_output([sys.executable,'-m','pytest','test/','--collect-only','-q'],cwd=ROOT,text=True)
    nodes={line for line in output.splitlines() if line.startswith('test/') and '::' in line}
    expected=set((ROOT/'test/expected_tests.txt').read_text().splitlines())
    assert nodes == expected


def test_website_local_links_and_assets():
    from bs4 import BeautifulSoup
    page=BeautifulSoup((ROOT/'docs/index.html').read_text(),'html.parser')
    ids={tag['id'] for tag in page.find_all(id=True)}
    for link in page.find_all('a',href=True):
        target=link['href']
        if target.startswith('#') and len(target)>1: assert target[1:] in ids
    for image in page.find_all('img'):
        assert image.get('alt') and (ROOT/'docs'/image['src']).is_file()
    assert not page.find_all('script')
    assert 'GATO' not in page.get_text()
    citation = page.find('pre', id='bibtex')
    assert citation and '@inproceedings{adabag2024mpcgpu' in citation.get_text()
    assert citation.find_parent('details') is None
    assert not citation.has_attr('hidden')
    assert 'MIT license' in page.footer.get_text()


def test_icra_fixture_provenance():
    # Verbatim paper-code files (examples/trajfiles/0_0_* at commits 077252e and c556b19).
    icra = ROOT / "examples/icra"
    digests = {"pick_place_traj.csv": "ca3a68cb3cc715f7f647f487ab1a9a812ed2a6f52297d24a36441a22f9fff3bb",
               "pick_place_eepos_2024.traj": "b01a0d151d4b449d61de94920cc3b873d31a387958a3ab5651d2b9284beb1199"}
    for name, digest in digests.items():
        assert hashlib.sha256((icra / name).read_bytes()).hexdigest() == digest, name
    import numpy as np
    current = np.loadtxt(icra / "pick_place_eepos.traj", delimiter=",")
    original = np.loadtxt(icra / "pick_place_eepos_2024.traj", delimiter=",")
    assert current.shape == original.shape == (666, 6)
    # The named flange frame sits 4.0 cm beyond the old link-7 origin at every configuration.
    np.testing.assert_allclose(np.linalg.norm(current[:, :3] - original[:, :3], axis=1), 0.04, atol=2e-4)
