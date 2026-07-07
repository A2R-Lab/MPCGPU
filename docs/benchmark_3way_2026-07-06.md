# 3-way iiwa14 fig8 benchmark — tracking + isolated per-solve timing (2026-07-06/07)

> **2026-07-07 UPDATE — the regularization pattern IS the preconditioner story (and the new
> benchmark config).** MPCGPU and GATO build the IDENTICAL block-tridiagonal stair
> preconditioner; what differed was the system it preconditions. Under MPCGPU's historic
> full-Q+R rho: cond(Pinv·S) ≈ 3.0e4 (raw S 3.7e7) → eta-exit lies ~500x, 200 uniform iters
> needed. Under GATO's reg (rho on the position half of Q only, R unregularized,
> `-DGATO_REG_PATTERN`): cond(Pinv·S) ≈ 2.1e2 — even though raw S is WORSE (9.2e8) — because
> bare R makes the theta diagonal blocks dominate S, which is exactly what a block-Jacobi
> stair captures. Result at native eta-exit: **avg 1.1 PCG iters/solve (max 23), 0.218 ms
> median/solve, tracking 0.0315/0.0744** (QDLDL cross-check 0.0324; memcheck clean; closed
> loop runs at rho≈0.55, accepted alpha≈1/8 — a more damped, GATO-like operating point).
> MPCGPU-PCG is now the fastest solver in the table at batch size 1. The earlier
> "reg pattern doesn't matter" test was confounded by the then-unfixed terminal-cost bug.
> run_3way_iiwa.sh now uses this config; the uniform-200 row below is kept for reference.

| solver (2026-07-07 config) | median / solve | tracking L2 mean / max |
|---|---|---|
| **MPCGPU (GPU), GBD-PCG + GATO_REG_PATTERN, native exit (avg 1.1 it)** | **0.218 ms** | 0.0315 / 0.0744 |
| GATO (GPU, batch=1) | 0.348 ms | 0.0323 / 0.0754 |
| MPCGPU (GPU), QDLDL | 0.384 ms | 0.0334 / 0.0753 |
| BatchThneed (CPU) | 3.06 ms | 0.0159 / 0.0727 |

All timing below is the 2026-07-06 measurement set (kept for reference; the uniform-200
MPCGPU-PCG row is superseded by the GATO_REG_PATTERN row above).

## Batch-size sweep (2026-07-07, isolated, same box)

MPCGPU has NO batch axis (one cooperative grid-wide solve owns the GPU per problem) — it is
the B=1 reference line. GATO sweep: `GATO examples/benchmarks/sweep_batch_iiwa_fig8.py`
(B identical replicas, open-loop warm-started over the same fig8 goal sequence, solver-internal
sqp_time_us, 400 solves/config). BT sweep: `track_iiwa_fig8_bt.py <sim_time> <B>` (closed loop,
B replicas, num_threads=B, 24-core box).

| B | GATO ms/solve | GATO us/traj | BT ms/solve | BT us/traj |
|---|---|---|---|---|
| 1 | 0.695* | 695 | 3.10 | 3101 |
| 2 | 0.841 | 421 | 3.26 | 1632 |
| 4 | 0.901 | 225 | 3.93 | 983 |
| 8 | 1.012 | 127 | 4.07 | 509 |
| 16 | 1.234 | 77 | 4.30 | 269 |
| 32 | 1.688 | 53 | 9.52 | 298 |
| 64 | 2.596 | 41 | 16.78 | 262 |
| 128 | 10.808 | 84 | 30.16 | 236 |

- GATO amortizes 17x to B=64 (41 us/traj), then falls off a cliff at B=128 (occupancy waves /
  cache pressure on this 170-SM part) — sweet spot B<=64 at N=64.
- BT saturates its 24 cores around B=16 (~250-270 us/traj plateau; per-solve latency keeps
  growing with oversubscription).
- (*) GATO's B=1 in the sweep driver (0.695 ms) is slower than in the closed-loop harness
  (0.348 ms): the open-loop driver's warm-start/rho trajectory differs from the sim-fed loop.
  Use 0.348 ms for the B=1 headline and the sweep for the SCALING shape (self-consistent
  across B). At B=64, GATO is ~6.4x BT per trajectory and ~5x MPCGPU-PCG's B=1 line
  (0.218 ms) in throughput terms.

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

The per-solve timing methodology is now scripted: `tools/time_persolve.sh [N] [pcg|qdldl]
[cycles] [out_csv]` (builds validate_track at KNOT_POINTS=N with the fair flags + SAVE_DATA,
median-of-run-medians). The paper Fig-3 data pipeline that consumes it lives in GATO:
`examples/paper-figures/reproduce_fig3_fair.py` (+ `sweep_batch_iiwa_fig8.py` N x B GATO
grid, `track_iiwa_fig8_bt.py <sim> <B> <N> <csv>`), CSVs in GATO
`examples/benchmarks/data/sweep_fig8_{gato,bt,mpcgpu}.csv`.

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
