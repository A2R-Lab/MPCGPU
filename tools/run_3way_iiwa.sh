#!/usr/bin/env bash
# FAIR 3-way iiwa14 fig8 tracking check (GATO / MPCGPU / BatchThneed) on the IDENTICAL problem:
#   robot iiwa14 (URDF eeb7d4ff), q0=readyC, EE=grid-L7, fig8 A=0.15 T=6 centered at L7(q0),
#   warm-start = zero controls, config = SQP=1 / PCG=200 / rel_tol=1e-4 / rho=0.01 / cost EE2 qd1e-2
#   u2e-6 N50 mu10. Tracking measured at the L7 frame for all three. NOT a timing run (functional).
# Usage: tools/run_3way_iiwa.sh [sim_time]   (run from MPCGPU repo root; GPU needed for GATO+MPCGPU)
set -uo pipefail
SIM=${1:-6.0}
MPCGPU=/home/plancher/Desktop/MPCGPU
GATO=/home/plancher/Desktop/GATO
GRIDVENV=/home/plancher/Desktop/GRiD/.venv
PY=$GRIDVENV/bin/python
LD=$MPCGPU/qdldl/build/out
CF="--compiler-options -Wall -O3 -DNDEBUG -arch=sm_120 -Iinclude -Iinclude/common -IGLASS -IGBD-PCG/include -lqdldl -Iqdldl/include -Lqdldl/build/out -lcublas"
FAIR="-DKNOT_POINTS=64 -DPCG_MAX_ITER=200 -DPCG_RES_TOL=1e-4 -DRHO_INIT=0.01 -DSQP_MAX_ITER=1 -DSQP_MAX_TIME_US=100000000"

cd "$MPCGPU" || exit 1
echo "==================== MPCGPU ===================="
echo "[build] gen_reference + validate_track (fair flags)"
nvcc $CF tools/gen_reference.cu   -o tools/gen_reference.exe   || { echo "gen_reference build FAILED"; }
nvcc $CF $FAIR tools/validate_track.cu -o tools/validate_track.exe || { echo "validate_track build FAILED"; }
echo "[gen] fig8 A=0.15 T=6 -> examples/trajfiles/0_0 (zero-control warm-start)"
LD_LIBRARY_PATH=$LD ./tools/gen_reference.exe examples/trajfiles/0_0 0.15 6 | sed 's/^/  /'
echo "[run] validate_track"
LD_LIBRARY_PATH=$LD ./tools/validate_track.exe examples/trajfiles/0_0 2>&1 | grep -E "RESULT|trace" | sed 's/^/  MPCGPU /'

echo "==================== GATO ===================="
PYTHONPATH=$GATO/python timeout 600 "$PY" "$GATO/examples/benchmarks/track_iiwa_fig8_gato.py" "$SIM" 2>&1 \
  | grep -viE "warn|deprecat" | grep -E "RESULT_GATO|trace|iiwa14 GATO" | sed 's/^/  /'

echo "==================== BatchThneed (CPU) ===================="
SQPCPU=$GATO/examples/benchmarks/baselines/sqpcpu; PREFIX=$SQPCPU/deps/install
CMEEL=$GRIDVENV/lib/python3.12/site-packages/cmeel.prefix/lib
LD_LIBRARY_PATH="$SQPCPU/build:$PREFIX/lib:$CMEEL:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$SQPCPU/build:$GATO/python:${PYTHONPATH:-}" \
  timeout 400 "$PY" "$GATO/examples/benchmarks/baselines/track_iiwa_fig8_bt.py" "$SIM" 2>&1 \
  | grep -viE "warn|deprecat" | grep -E "RESULT_BT|trace|iiwa14 BatchThneed" | sed 's/^/  /'

echo "==============================================="
echo "Compare L7_mean/max/final across the three (all track the identical fig8 at the L7 frame)."
