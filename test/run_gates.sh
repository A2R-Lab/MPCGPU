#!/usr/bin/env bash
# Gate-runner for GBD-PCG correctness (build + run, PASS/FAIL per gate). CORRECTNESS ONLY —
# not a timing harness. Needs a cooperative-launch-capable GPU. Sources live in ../examples.
#
# Gates:
#   1. test_pcg_spd   — random SPD block-tridiagonal S + identity Pinv, host residual check
#                       (the real solver gate; prints PASS/FAIL itself). Run at 6x8 and at
#                       MPCGPU's 14-state dims.
#   2. test_bdmv      — cooperative bdmv (glass::gemv strip matvec + loadbdVec halo +
#                       zeroed boundary strips) vs a host block-tridiagonal multiply.
#                       Uses the REAL dumped /tmp/mpc_S.bin if present (from an MPCGPU
#                       validate_track built with -DDUMP_KKT), else a synthetic random
#                       strip file — any strips are valid, it is a pure matvec check.
#   3. test_pcg_dumped— GBD-PCG standalone on the REAL dumped Schur system
#                       /tmp/mpc_{S,Pinv,gamma}.bin (float32 [L|D|R] strips). SKIPPED if
#                       the dumps are absent (produce them with an MPCGPU -DDUMP_KKT run).
#                       SMOKE gate only: the achievable true-residual plateau depends on
#                       the dump's regularization pattern (GATO_REG_PATTERN dumps converge
#                       to ~1e-5 rel; historic full-Q+R can plateau near 1e-2 — see the
#                       eta-exit notes in CLAUDE.md), so the bar is rel < 1e-1 and finite.
#
# Usage: test/run_gates.sh            (run from the GBD-PCG repo root)
#        ARCH=sm_86 test/run_gates.sh
set -uo pipefail
ARCH=${ARCH:-sm_120}
CF="--compiler-options -Wall -O3 -Iinclude -IGLASS -arch=$ARCH"
[[ -f include/pcg.cuh ]] || { echo "run from the GBD-PCG repo root"; exit 1; }

npass=0; nfail=0; nskip=0
gate(){ if [[ "$2" == "0" ]]; then echo "PASS  $1  $3"; npass=$((npass+1));
        else echo "FAIL  $1  $3"; nfail=$((nfail+1)); fi; }

# ---------------- gate 1: test_pcg_spd at two dim configs ------------------------------
for dims in "6 8" "14 32"; do
  read -r SS KP <<<"$dims"
  echo "[gate 1] build test_pcg_spd STATE_SIZE=$SS KNOT_POINTS=$KP"
  if nvcc $CF -DSTATE_SIZE=$SS -DKNOT_POINTS=$KP examples/test_pcg_spd.cu -o examples/gates_spd_${SS}x${KP}.exe; then
    out=$(./examples/gates_spd_${SS}x${KP}.exe 2>&1); rc=$?; echo "$out" | sed 's/^/    /'
    gate "test_pcg_spd ${SS}x${KP}" "$rc" "exit code + self-reported PASS (rel res < 1e-4)"
  else
    gate "test_pcg_spd ${SS}x${KP}" 1 "build failed"
  fi
done

# ---------------- gate 2: test_bdmv (matvec vs host reference) -------------------------
SS=14
if [[ -f /tmp/mpc_S.bin ]]; then
  SFILE=/tmp/mpc_S.bin
  KP=$(python3 -c "import os; print(os.path.getsize('$SFILE')//4//(3*$SS*$SS))")
  echo "[gate 2] using dumped $SFILE (KNOT_POINTS=$KP)"
else
  KP=32; SFILE=/tmp/gates_bdmv_S.bin
  echo "[gate 2] no /tmp/mpc_S.bin; synthesizing random strips ($SFILE, ${SS}x${KP})"
  python3 -c "
import numpy as np
np.random.seed(0)
np.random.uniform(-1, 1, 3*$SS*$SS*$KP).astype(np.float32).tofile('$SFILE')"
fi
echo "[gate 2] build test_bdmv STATE_SIZE=$SS KNOT_POINTS=$KP"
if nvcc $CF -DSTATE_SIZE=$SS -DKNOT_POINTS=$KP examples/test_bdmv.cu -o examples/gates_bdmv.exe; then
  out=$(./examples/gates_bdmv.exe "$SFILE" 2>&1)
  grep -E "matrix=|max\|diff\|" <<<"$out" | head -3 | sed 's/^/    /'
  rel=$(grep -oE "rel [0-9.eE+-]+" <<<"$out" | head -1 | awk '{print $2}' || echo nan)
  ok=$(python3 -c "r=float('$rel'); print(0 if r < 1e-4 else 1)" 2>/dev/null || echo 1)
  gate "test_bdmv          " "$ok" "GPU vs host matvec rel=$rel (bar 1e-4; float32 noise ~1e-7)"
else
  gate "test_bdmv          " 1 "build failed"
fi

# ---------------- gate 3: test_pcg_dumped (real Schur system, smoke) --------------------
if [[ -f /tmp/mpc_S.bin && -f /tmp/mpc_Pinv.bin && -f /tmp/mpc_gamma.bin ]]; then
  KP=$(python3 -c "import os; print(os.path.getsize('/tmp/mpc_S.bin')//4//(3*$SS*$SS))")
  echo "[gate 3] build test_pcg_dumped STATE_SIZE=$SS KNOT_POINTS=$KP"
  if nvcc $CF -DSTATE_SIZE=$SS -DKNOT_POINTS=$KP examples/test_pcg_dumped.cu -o examples/gates_dumped.exe; then
    out=$(./examples/gates_dumped.exe 1e-4 500 2>&1); echo "$out" | sed 's/^/    /'
    rel=$(grep -oE "\|\|gamma\|\| = [0-9.eE+-]+" <<<"$out" | grep -oE "[0-9.eE+-]+$" || echo nan)
    ok=$(python3 -c "r=float('$rel'); print(0 if r < 1e-1 else 1)" 2>/dev/null || echo 1)
    gate "test_pcg_dumped   " "$ok" "true rel residual=$rel (smoke bar 1e-1; see header note)"
  else
    gate "test_pcg_dumped   " 1 "build failed"
  fi
else
  echo "SKIP  test_pcg_dumped    /tmp/mpc_{S,Pinv,gamma}.bin absent (dump via MPCGPU -DDUMP_KKT run)"
  nskip=$((nskip+1))
fi

echo "==============================================="
echo "gates: $npass PASS, $nfail FAIL, $nskip SKIP"
exit $(( nfail > 0 ))
