#!/usr/bin/env bash
# Public correctness entry point; build flags and gate outcomes have one owner.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" -m pytest test/test_gates.py -q "$@"
