#!/usr/bin/env bash
# Isolated MPCGPU per-solve timing at KNOT_POINTS=N on the FAIR iiwa14 fig8 problem
# (2026-07-07 benchmark config: GATO_REG_PATTERN + native eta-exit — see docs/benchmark_3way_2026-07-06.md).
# Scripts the methodology used for the 3-way table: build validate_track with -DSAVE_DATA=1,
# run `cycles` closed-loop passes, take the MEDIAN-of-run-medians of the per-solve sqp times.
# MPCGPU has NO batch axis (one cooperative grid-wide solve owns the GPU) -> this is the B=1 line;
# a batch of M problems costs M x this number (sequential).
#
# Usage: tools/time_persolve.sh [N=64] [pcg|qdldl=pcg] [cycles=3] [out_csv]
#   out_csv (optional): append "N,1,median_ms,p90_ms,per_traj_us,n_solves,L2_mean" (fig3 schema).
# Run from the MPCGPU repo root on a QUIET box (standing rule: nothing else on CPU/GPU).
set -uo pipefail
N=${1:-64}
LINSYS=${2:-pcg}
CYCLES=${3:-3}
OUT=${4:-}
[[ "$LINSYS" == "pcg" ]] && LS=1 || LS=0
LD=qdldl/build/out
CF="--compiler-options -Wall -O3 -DNDEBUG -arch=sm_120 -Iinclude -Iinclude/common -IGRiD/grid_codegen/collision -IGLASS -IGBD-PCG/include -lqdldl -Iqdldl/include -Lqdldl/build/out -lcublas"
FAIR="-DKNOT_POINTS=$N -DPCG_MAX_ITER=200 -DPCG_RES_TOL=1e-4 -DGATO_REG_PATTERN -DRHO_INIT=0.01 -DSQP_MAX_ITER=1 -DSQP_MAX_TIME_US=100000000"
EXE=tools/vt_time_${LINSYS}_N${N}.exe

if [[ ! -f examples/trajfiles/0_0_eepos.traj ]]; then
  echo "[time_persolve] generating fig8 trajfiles (A=0.15 T=6)"
  nvcc $CF tools/gen_reference.cu -o tools/gen_reference.exe || exit 1
  LD_LIBRARY_PATH=$LD ./tools/gen_reference.exe examples/trajfiles/0_0 0.15 6 || exit 1
fi

echo "[time_persolve] build $EXE (KNOT_POINTS=$N LINSYS=$LINSYS, fair flags + SAVE_DATA)"
nvcc $CF $FAIR -DLINSYS_SOLVE=$LS -DSAVE_DATA=1 tools/validate_track.cu -o "$EXE" || exit 1

# stdlib-only stats (the system python3 has no numpy); values pass as argv, never
# interpolated into python source. p90 = nearest-rank on the sorted samples.
stats_of_file() { python3 -c '
import sys, statistics as st
v = sorted(float(x) for x in open(sys.argv[1]).read().split())
print(f"{st.median(v):.1f} {v[round(0.9 * (len(v) - 1))]:.1f} {len(v)}")' "$1"; }
median_of() { python3 -c '
import sys, statistics as st
print(f"{st.median(float(x) for x in sys.argv[1:]):.1f}")' "$@"; }

medians=(); p90s=(); ns=(); track=""
for c in $(seq 1 "$CYCLES"); do
  out=$(LD_LIBRARY_PATH=$LD ./"$EXE" examples/trajfiles/0_0 2>&1)
  line=$(grep -E "RESULT" <<<"$out" | tail -1)
  track=$(grep -oE "mean=[0-9.eE+-]+" <<<"$line" | head -1 | cut -d= -f2 || true)
  stats=$(stats_of_file tmp/results/validate_0_sqp_times.result) || { echo "[time_persolve] ERROR: no per-solve times (SAVE_DATA output missing?)"; exit 1; }
  read -r med p90 n <<<"$stats"
  medians+=("$med"); p90s+=("$p90"); ns+=("$n")
  echo "  cycle $c: median=${med}us p90=${p90}us n=${n}  ${line}"
done

MED=$(median_of "${medians[@]}") || exit 1
P90=$(median_of "${p90s[@]}") || exit 1
echo "RESULT_MPCGPU N=$N linsys=$LINSYS median_us=$MED p90_us=$P90 (median of $CYCLES run-medians)"

if [[ -n "$OUT" ]]; then
  [[ -f "$OUT" ]] || echo "N,B,median_ms,p90_ms,per_traj_us,n_solves,L2_mean" > "$OUT"
  python3 -c '
import sys
n, med, p90, ns, tr = sys.argv[1:6]
print(f"{n},1,{float(med)/1000:.4f},{float(p90)/1000:.4f},{med},{ns},{tr}")' \
    "$N" "$MED" "$P90" "${ns[-1]}" "${track:-nan}" >> "$OUT" || exit 1
  echo "[time_persolve] appended row -> $OUT"
fi
