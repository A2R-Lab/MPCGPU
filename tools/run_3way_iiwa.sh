#!/usr/bin/env bash
# Historical integration entry point: never launch mutable sibling projects.
set -euo pipefail
echo 'The old three-project runner is retired. MPCGPU needs no GATO checkout.' >&2
echo 'Use tools/run_gates.sh for local correctness; see docs/timing.md for timing.' >&2
exit 2
