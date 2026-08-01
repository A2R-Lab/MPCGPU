#!/usr/bin/env bash
# Single gate-runner for MPCGPU correctness (build + run, PASS/FAIL per gate).
# CORRECTNESS ONLY — never read timing off these runs (standing rule: timing needs an
# isolated quiet box; correctness runs may share the box, but builds here are sequential
# on purpose — keep it that way, no parallel nvcc).
#
# Gates, in order (expected references from docs/benchmark_3way_2026-07-06.md, fair
# 2026-07-07 config: SQP=1, PCG cap 200, rel tol 1e-4, RHO_INIT=0.01, GATO_REG_PATTERN):
#   1. single_cost_test    — ONE sqpSolvePcg from the hold warm-start responds to the EE
#                            cost: ||d_xu step|| ~= 339.6 AND post-solve window tracking
#                            ~= 0.0107 (both bit-repeatable at the fair config, measured
#                            2026-07-08; the step is large because GATO_REG_PATTERN leaves
#                            R unregularized. Bug signature: ~0 step / post == pre ~0.064.
#                            The old 0.446 reference was a different flag combo — see the
#                            2026-07-06 baseline note — do not resurrect it).
#   2. test_terminal_cost  — terminal-knot q-gradient, 4 kernels vs the pinocchio truth
#                            vector hardcoded in tools/test_terminal_cost.cu
#                            ([0.0026, 0.1289, 0.0020, -0.3513, 0.0002, 0.0729, 0], w=50,
#                            for the COMMITTED tools/data/ dump inputs).
#                            Regression test for the kkt.cuh terminal-cost smem aliasing
#                            bug (s_Qk offset 6 -> 2*6, fixed in 88c3853).
#   3. validate_track PCG  — closed-loop fig8 tracking, GBD-PCG linsys + GATO_REG_PATTERN:
#                            expect mean ~0.0288 (max ~0.0635; contact-frame EE, GRiD e31f7bd).
#   4. validate_track QDLDL— same problem, QDLDL linsys: expect mean ~0.0294 (max ~0.0649).
#
# Usage: tools/run_gates.sh          (run from the MPCGPU repo root; needs a GPU +
#                                     qdldl built: make build_qdldl)
#        ARCH=sm_86 tools/run_gates.sh   (override GPU arch)
#        GATES_REDUMP=1 tools/run_gates.sh  (re-dump fresh gate-2 inputs to /tmp — only for
#                                            RE-PINNING the gate; see the gate-2 note below)
set -uo pipefail
ARCH=${ARCH:-sm_120}
LD=qdldl/build/out
CF="--compiler-options -Wall -O3 -DNDEBUG -arch=$ARCH -Iinclude -Iinclude/common -IGRiD/grid_codegen/collision -IGLASS -IGBD-PCG/include -lqdldl -Iqdldl/include -Lqdldl/build/out -lcublas"
FAIR="-DKNOT_POINTS=64 -DPCG_MAX_ITER=200 -DPCG_RES_TOL=1e-4 -DGATO_REG_PATTERN -DRHO_INIT=0.01 -DSQP_MAX_ITER=1 -DSQP_MAX_TIME_US=100000000"

[[ -f include/common/settings.cuh ]] || { echo "run from the MPCGPU repo root"; exit 1; }
[[ -f $LD/libqdldl.so ]] || { echo "qdldl not built: make build_qdldl"; exit 1; }
mkdir -p tmp/results

npass=0; nfail=0
gate(){ # gate <name> <0=pass,else fail> <detail>
  if [[ "$2" == "0" ]]; then echo "PASS  $1  $3"; npass=$((npass+1));
  else echo "FAIL  $1  $3"; nfail=$((nfail+1)); fi
}

# prereq: fig8 trajfiles (A=0.15 T=6, same recipe as run_3way_iiwa.sh / time_persolve.sh)
if [[ ! -f examples/trajfiles/0_0_eepos.traj ]]; then
  echo "[gates] generating fig8 trajfiles (A=0.15 T=6)"
  nvcc $CF tools/gen_reference.cu -o tools/gen_reference.exe || exit 1
  LD_LIBRARY_PATH=$LD ./tools/gen_reference.exe examples/trajfiles/0_0 0.15 6 || exit 1
fi

# ---------------- gate 1: single_cost_test (one-solve EE-cost response) ----------------
echo "[gate 1] build single_cost_test (fair flags)"
if nvcc $CF $FAIR tools/single_cost_test.cu -o tools/gates_sct.exe; then
  out=$(LD_LIBRARY_PATH=$LD ./tools/gates_sct.exe examples/trajfiles/0_0 2>&1)
  step=$(grep -oE "d_xu_before \|\| = [0-9.eE+-]+" <<<"$out" | grep -oE "[0-9.eE+-]+$" || echo nan)
  post=$(grep -oE "\[post\] mean \|EE_k - goal_k\| over window = [0-9.eE+-]+" <<<"$out" | grep -oE "[0-9.eE+-]+$" || echo nan)
  echo "$out" | grep -E "EE_COST|mean \|EE_k" | sed 's/^/    /'
  ok=$(python3 -c "s=float('$step'); p=float('$post'); print(0 if 300 <= s <= 380 and 0.005 <= p <= 0.02 else 1)" 2>/dev/null || echo 1)
  gate "single_cost_test " "$ok" "||d_xu step||=$step (expect ~339.6; band 300-380) post-window=$post (expect ~0.0107; band 0.005-0.02)"
else
  gate "single_cost_test " 1 "build failed"
fi

# ---------------- gate 2: test_terminal_cost (terminal q-gradient vs pin truth) --------
# Inputs are COMMITTED: tools/data/mpc_{xu_pre,goal}.bin (solve-3000 dump from a fair-config
# closed loop, 2026-07-07) — the hardcoded pin-truth vector in test_terminal_cost.cu was
# computed for exactly those files, so the gate is /tmp- and re-dump-independent.
# GATES_REDUMP=1 dumps FRESH inputs to /tmp/mpc_*.bin (DUMP_KKT build) — that is for
# RE-PINNING the gate only: copy them to tools/data/ AND recompute the truth vector with
# pinocchio (recipe in test_terminal_cost.cu's header), or the gate will rightly fail.
if [[ "${GATES_REDUMP:-0}" == "1" ]]; then
  echo "[gate 2] dumping fresh solve-3000 inputs to /tmp (re-pin: copy to tools/data/ + recompute the truth)"
  nvcc $CF $FAIR -DDUMP_KKT -DDUMP_KKT_AT_SOLVE=3000 tools/validate_track.cu -o tools/gates_vt_dump.exe \
    && LD_LIBRARY_PATH=$LD ./tools/gates_vt_dump.exe examples/trajfiles/0_0 > /dev/null 2>&1
fi
echo "[gate 2] build test_terminal_cost (build line from its header; KNOT_POINTS baked at N=64 in main)"
if nvcc --compiler-options -Wall -O3 -DNDEBUG -arch=$ARCH -Iinclude -Iinclude/common -IGRiD/grid_codegen/collision -IGLASS \
        -IGBD-PCG/include -Iqdldl/include tools/test_terminal_cost.cu -o tools/gates_tct.exe -lcublas; then
  out=$(./tools/gates_tct.exe 2>&1); echo "$out" | sed 's/^/    /'
  ok=$(TCT_OUT="$out" python3 - <<'EOF'
import os, re
rows = {}
for ln in os.environ["TCT_OUT"].splitlines():
    m = re.match(r"\((\w+)\).*q-block:((\s+-?[0-9.]+)+)", ln)
    if m: rows[m.group(1)] = [float(x) for x in m.group(2).split()]
    m = re.match(r"\(pin truth.*\):((\s+-?[0-9.]+)+)", ln)
    if m: rows["truth"] = [float(x) for x in m.group(1).split()]
need = ["A", "B", "C", "D", "truth"]
if any(k not in rows for k in need): print(1); raise SystemExit
mut  = max(abs(a-b) for k in "BCD" for a, b in zip(rows["A"], rows[k]))
tru  = max(abs(a-b) for a, b in zip(rows["A"], rows["truth"]))
# aliasing-bug signature is huge (dumped bad g row: -3.27 vs truth 6.31) -> 0.05 abs is ample
print(0 if (mut < 0.05 and tru < 0.05) else 1)
EOF
)
  gate "test_terminal_cost" "$ok" "A/B/C/D mutual + vs pin truth, tol 0.05 abs"
else
  gate "test_terminal_cost" 1 "build failed"
fi

# ---------------- gates 3+4: validate_track, one pass per linsys ------------------------
run_track(){ # run_track <gate#> <tag> <extra nvcc flags> <exe> <expected mean>
  echo "[gate $1] build validate_track ($2)"
  if nvcc $CF $FAIR $3 tools/validate_track.cu -o "$4"; then
    out=$(LD_LIBRARY_PATH=$LD ./"$4" examples/trajfiles/0_0 2>&1)
    line=$(grep -E "^RESULT" <<<"$out" | tail -1); echo "    $line"
    mean=$(grep -oE "mean=[0-9.eE+-]+" <<<"$line" | cut -d= -f2 || echo nan)
    ok=$(python3 -c "m=float('$mean'); print(0 if 0.02 <= m <= 0.05 else 1)" 2>/dev/null || echo 1)
    gate "validate_$2" "$ok" "tracking mean=$mean (expect ~$5; band 0.02-0.05; divergence = 0.19+/NaN)"
  else
    gate "validate_$2" 1 "build failed"
  fi
}
run_track 3 "track_pcg  " ""                "tools/gates_vt_pcg.exe"   0.0288
run_track 4 "track_qdldl" "-DLINSYS_SOLVE=0" "tools/gates_vt_qdldl.exe" 0.0294

echo "==============================================="
echo "gates: $npass PASS, $nfail FAIL"
exit $(( nfail > 0 ))
