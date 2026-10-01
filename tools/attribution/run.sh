#!/usr/bin/env bash
# TIMING (exclusive quiet window only): run the prepared attribution binaries from prepare.sh.
# Writes tmp/attribution/results-<timestamp>/ : kernel benchmark lines and paper-era MPC outputs.
set -euo pipefail
[[ "${MPCGPU_QUIET_WINDOW:-}" == 1 ]] || { echo "REFUSED: set MPCGPU_QUIET_WINDOW=1 in an assigned slot" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/tmp/attribution
HORIZONS=${HORIZONS:-"32 64 128 256 512"}
REPEATS=${REPEATS:-3}
cd "$OUT/bin" && sha256sum -c --quiet SHA256SUMS
RES=$OUT/results-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RES"
quiet() {  # GATO's test (GPU util <= 5%, no compute apps, load <= 2); waits up to 2 min to settle
    local util apps load waited=0
    while :; do
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
        load=$(awk '{print int($1)}' /proc/loadavg)
        (( util <= 5 && apps == 0 && load <= 2 )) && return 0
        (( waited >= 120 )) && { echo "NOT QUIET util=$util apps=$apps load=$load" >&2; exit 3; }
        sleep 5; waited=$((waited + 5))
    done
}
{ date -u; nvidia-smi; nvcc --version; } > "$RES/environment.txt"
export LD_LIBRARY_PATH=$ROOT/qdldl/build/out:${LD_LIBRARY_PATH:-}

# 1. Kernel benchmark: identical captured systems, three kernel versions, alternating order.
for repeat in $(seq 1 "$REPEATS"); do
    versions="paper glass current"; (( repeat % 2 == 0 )) && versions="current glass paper"
    for n in $HORIZONS; do
        for version in $versions; do
            quiet
            for spec in "fixed 1" "fixed 25" "fixed 100" "tol 1e-4"; do
                set -- $spec
                echo "repeat=$repeat version=$version $("$OUT/bin/bench-$version-$n.exe" "$OUT/dumps/N$n" "$1" "$2" 300)" \
                    >> "$RES/kernel.txt"
            done
        done
    done
done

# 2. Paper-era MPC code as published (wall-clock 2000 us SQP budget, TIME_LINSYS=1), run from its tree.
for repeat in $(seq 1 "$REPEATS"); do
    for n in $HORIZONS; do
        for backend in pcg qdldl; do
            quiet
            dir=$RES/paper/r$repeat
            mkdir -p "$dir/tmp/results"
            (cd "$dir" && ln -sfn "$OUT/paper-code/examples" examples \
                && "$OUT/bin/paper-$backend-$n.exe" > "$dir/$backend-$n.log" 2>&1)
        done
    done
done
echo "DONE $RES"
