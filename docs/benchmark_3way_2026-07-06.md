# 3-way iiwa14 fig8 benchmark — tracking + isolated per-solve timing (2026-07-06)

Identical problem for all solvers: iiwa14 (URDF eeb7d4ff), q0=readyC, EE frame = grid-L7,
EE-space figure-8 A=0.15 T=6s centered at L7(q0), dt=0.01, N=64 knots, SQP=1 (real-time
iteration), rho=0.01, cost EE=2 qd=1e-2 u=2e-6 N=50 qlim=0.01, mu=10, zero-control warm start,
one-stage shift. Tracking = L2 error at the L7 frame. Timing = isolated runs on a quiet
RTX 5090 box, one solver at a time, 3 cycles each (median-of-medians; per-run p10-p90 spread
< 1%). Solve time = full SQP-1 solve wall time (KKT + Schur + linear solve + dz + 8-alpha
line search) as measured by each harness.

| solver | linear solve | median / solve | tracking L2 mean / max |
|---|---|---|---|
| GATO (GPU, batch=1) | single-block `glass::pcg` (adaptive, ~25-70 it) | **0.348 ms** | 0.0323 / 0.0754 |
| MPCGPU (GPU) | QDLDL (host LDLT) | 0.384 ms | 0.0334 / 0.0753 |
| MPCGPU (GPU) | cooperative GBD-PCG, uniform 200 it | 1.155 ms | 0.0326 / 0.0753 |
| BatchThneed (CPU) | OSQP | 3.06 ms | 0.0159 / 0.0727 |

Runs: `tools/run_3way_iiwa.sh` (functional), `tools/vt_time_{pcg,qdldl}.exe` built with
`-DSAVE_DATA=1` + FAIR flags (per-solve times in tmp/results/validate_0_sqp_times.result);
GATO `examples/benchmarks/track_iiwa_fig8_gato.py`, BT `track_iiwa_fig8_bt.py` (12s runs).

Notes / caveats:
- **MPCGPU-PCG runs a UNIFORM 200 iterations/solve** (PCG_RES_TOL=0): its eta-recurrence exit
  under-reports the true residual ~500x on this system and fires ~10x early (tracking 0.19 vs
  0.033); even true-residual-certified adaptive exits under-deliver vs uniform iterations
  (residual bounds ||r||, not the low-eigenmode lambda error the long-horizon loop feels;
  preconditioned cond ~3e4). With its native exit MPCGPU-PCG would be ~0.35 ms/solve but
  6x worse tracking — not a defensible operating point. See GBD-PCG
  PCG_TRUE_EXIT_CHECK_PERIOD (off by default) for the instrumented adaptive-exit experiment.
- A better preconditioner is the real future fix (fewer iterations AND honest exits).
- Cadence differs by design: MPCGPU solves every 2 ms sim time (5 solves per 10 ms knot
  step; duty at 1.155 ms/solve ≈ 58%), GATO/BT solve once per 10 ms step (GATO duty ≈ 3.5%).
- GATO tracking mean over 12 s (two fig8 periods) = 0.0259; BT = 0.0134. The 6 s means above
  match the shared run_3way output for consistency.
- All numbers post terminal-cost fix (88c3853) + L2 metric fix (af8f038). Numbers predating
  2026-07-06 used an L1 metric and carried the terminal-cost bug — do not mix.
