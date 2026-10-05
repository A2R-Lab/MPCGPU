#!/usr/bin/env bash
# Full, clean-source correctness receipt. No timing or dependency installation.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 0 || -n "${PYTEST_ADDOPTS:-}" ]]; then
    echo "Receipt must cover the full suite; selection arguments/PYTEST_ADDOPTS are forbidden." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --ignore-submodules=untracked)" ]]; then
    echo "Commit source changes before signing a receipt." >&2
    exit 1
fi
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" -c 'import importlib.metadata as m; assert m.version("pytest-gpu-proof") == "0.4.0", "Install requirements-dev.txt first"; import numpy, sympy, bs4, lxml, yaml, pinocchio'
[[ -f qdldl/build/out/libqdldl.so ]] || make build_qdldl
"$PYTHON" -m pytest test/ -q --gpu-proof-enable --gpu-proof-out gpu-proof.json --gpu-proof-github-user plancherb1
