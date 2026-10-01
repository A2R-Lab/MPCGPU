#!/usr/bin/env bash
# Historical integration entry point: never launch mutable sibling projects.
set -euo pipefail
echo 'The old three-project runner is retired. Use the self-contained MPCGPU gates.' >&2
echo 'Use tools/run_gates.sh for local correctness; see the timing section of docs/development.md.' >&2
exit 2
