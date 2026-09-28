#!/usr/bin/env bash
# Never compile inside a timing interval or consume a previous run's output.
set -euo pipefail
echo 'Legacy timing launcher retired. Use tools/timing.py prepare, then run --dry-run.' >&2
echo 'See docs/timing.md; actual run requires an explicitly assigned quiet window.' >&2
exit 2
