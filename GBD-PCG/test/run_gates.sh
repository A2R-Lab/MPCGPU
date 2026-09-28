#!/usr/bin/env bash
# GBD-PCG is in-tree; use the same individually attested gates as MPCGPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-.venv/bin/python}" -m pytest test/test_gates.py -q -k gbd "$@"
