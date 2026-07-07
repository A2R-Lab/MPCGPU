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
CF="--compiler-options -Wall -O3 -DNDEBUG -arch=sm_120 -Iinclude -Iinclude/common -IGLASS -IGBD-PCG/include -lqdldl -Iqdldl/include -Lqdldl/build/out -lcublas"
FAIR="-DKNOT_POINTS=$N -DPCG_MAX_ITER=200 -DPCG_RES_TOL=1e-4 -DGATO_REG_PATTERN -DRHO_INIT=0.01 -DSQP_MAX_ITER=1 -DSQP_MAX_TIME_US=100000000"
EXE=tools/vt_time_${LINSYS}_N${N}.exe

if [[ ! -f examples/trajfiles/0_0_eepos.traj ]]; then
  echo "[time_persolve] generating fig8 trajfiles (A=0.15 T=6)"
  nvcc $CF tools/gen_reference.cu -o tools/gen_reference.exe || exit 1
  LD_LIBRARY_PATH=$LD ./tools/gen_reference.exe examples/trajfiles/0_0 0.15 6 || exit 1
fi

echo "[time_persolve] build $EXE (KNOT_POINTS=$N LINSYS=$LINSYS, fair flags + SAVE_DATA)"
nvcc $CF $FAIR -DLINSYS_SOLVE=$LS -DSAVE_DATA=1 tools/validate_track.cu -o "$EXE" || exit 1

medians=(); p90s=(); ns=(); track=""
for c in $(seq 1 "$CYCLES"); do
  out=$(LD_LIBRARY_PATH=$LD ./"$EXE" examples/trajfiles/0_0 2>&1)
  line=$(grep -E "RESULT" <<<"$out" | tail -1)
  track=$(grep -oE "mean=[0-9.eE+-]+" <<<"$line" | head -1 | cut -d= -f2 || true)
  stats=$(python3 - <<'EOF'
import numpy as np
t = np.loadtxt("tmp/results/validate_0_sqp_times.result")
print(f"{np.median(t):.1f} {np.percentile(t,90):.1f} {len(t)}")
EOF
)
  read -r med p90 n <<<"$stats"
  medians+=("$med"); p90s+=("$p90"); ns+=("$n")
  echo "  cycle $c: median=${med}us p90=${p90}us n=${n}  ${line}"
done

read -r MED P90 <<<"$(python3 -c "
import numpy as np
m = np.array('${medians[*]}'.split(), float); p = np.array('${p90s[*]}'.split(), float)
print(f'{np.median(m):.1f} {np.median(p):.1f}')")"
echo "RESULT_MPCGPU N=$N linsys=$LINSYS median_us=$MED p90_us=$P90 (median of $CYCLES run-medians)"

if [[ -n "$OUT" ]]; then
  [[ -f "$OUT" ]] || echo "N,B,median_ms,p90_ms,per_traj_us,n_solves,L2_mean" > "$OUT"
  python3 -c "print(f'$N,1,{$MED/1000:.4f},{$P90/1000:.4f},$MED,${ns[-1]},${track:-nan}')" >> "$OUT"
  echo "[time_persolve] appended row -> $OUT"
fi
