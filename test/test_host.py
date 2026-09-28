"""CUDA-free contract gates, also run in ordinary CI."""
import hashlib
import importlib.util
import json
import os
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
    result = subprocess.run(["bash", "test/run_gpu_proof.sh", "-k", "one_test"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "full suite" in result.stderr


def test_expected_collection_manifest():
    output=subprocess.check_output([sys.executable,'-m','pytest','test/','--collect-only','-q'],cwd=ROOT,text=True)
    nodes={line for line in output.splitlines() if line.startswith('test/') and '::' in line}
    expected=set((ROOT/'test/expected_tests.txt').read_text().splitlines())
    assert nodes == expected


def test_website_local_links_and_assets():
    from bs4 import BeautifulSoup
    page=BeautifulSoup((ROOT/'website/index.html').read_text(),'html.parser')
    ids={tag['id'] for tag in page.find_all(id=True)}
    for link in page.find_all('a',href=True):
        target=link['href']
        if target.startswith('#') and len(target)>1: assert target[1:] in ids
    for image in page.find_all('img'):
        assert image.get('alt') and (ROOT/'website'/image['src']).is_file()
    assert not page.find_all('script')
