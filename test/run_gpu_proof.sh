#!/usr/bin/env bash
# Run the MPCGPU correctness-gate suite and emit a SIGNED gpu-proof receipt
# (gpu-proof.json). Mirrors GATO's GPU CI pattern: the suite runs on the lab
# GPU box, signs a receipt binding {git SHA, source fingerprint, per-gate
# outcomes, GPU info}, and the CPU-only GitHub Action
# (.github/workflows/verify-gpu-proof.yml) verifies the signature against
# github.com/plancherb1.keys on every push. Commit the receipt together with
# (or right after) the change it attests.
#
# Prerequisites:
#   - GPU box; submodules checked out (make submodules)
#   - a sibling GATO checkout (../GATO, or GATO_ROOT=) for the grid.cuh
#     byte-diff gate
#   - an SSH signing key (~/.ssh/id_*) whose public half is on the
#     keyholder's GitHub
#
# Usage:
#   ./test/run_gpu_proof.sh                      # full receipt -> gpu-proof.json
#   PYTHON=path/to/python ./test/run_gpu_proof.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Refuse to sign a dirty tree: the fingerprint cannot descend into the
# submodule gitlinks; a clean tree is what pins them via the receipt's commit
# SHA (mirrors test/gpu-proof-policy.yaml allow_dirty:false). Untracked content
# inside a submodule / ignored .exe debris is fine — the pins are what matter.
if [[ -n "$(git status --porcelain --ignore-submodules=untracked)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before signing a receipt." >&2
    exit 1
fi

# qdldl is a gate prerequisite (validate_track QDLDL arm links it)
[[ -f qdldl/build/out/libqdldl.so ]] || make build_qdldl

# Bootstrap a local venv if no python is provided (MPCGPU has no python
# package of its own — the venv exists only for pytest + the receipt plugin).
if [[ -z "${PYTHON:-}" ]]; then
    [[ -d .venv ]] || python3 -m venv .venv
    PYTHON=.venv/bin/python
fi

"$PYTHON" -m pip install -q "pytest-gpu-proof>=0.1" pytest pyyaml

# --gpu-proof-github-user: the signer must be the human KEYHOLDER — the
# plugin's remote-derived default would guess the org (A2R-Lab), and orgs have
# no SSH keys. The rest of the config lives in pyproject [tool.gpu_proof].
"$PYTHON" -m pytest test/ -q "$@" \
    --gpu-proof-enable \
    --gpu-proof-out gpu-proof.json \
    --gpu-proof-github-user plancherb1

echo
echo "Signed receipt: gpu-proof.json — 'git add gpu-proof.json' to attest this run."
