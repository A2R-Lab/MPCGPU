# 3-way iiwa14 fig8 benchmark — tracking + isolated per-solve timing (2026-08-01)

Supersedes `benchmark_3way_2026-07-06.md` (kept for the methodology history and
the regularization-pattern/preconditioner analysis, which stands unchanged).
**Every number in the old doc is stale**: the 2026-07-30 named-target regen
moved the solver cost, the reference trajfiles, and the tracking metric from
the L7 link frame to the URDF "EE" fixed joint (+0.04 m), and the GLASS
a3fa160 / GRiD e31f7bd bump shifted dynamics ULPs besides. Do not mix eras.

Provenance: ONE serial quiet-box run, `GATO examples/benchmarks/
run_timing_night.sh` → `night_logs/20260801_035038` (GATO @dbb5ec3, MPCGPU
@05268e4; RTX 5090; preflight recorded: GPU 5%, 343 MiB, load 0.21; correctness
fence 4/4 MPCGPU gates + 57 GATO gpu-pytests green in the same run).

## Headline: closed-loop 3-way parity (identical problem, EE frame)

iiwa14 (URDF eeb7d4ff), q0=readyC, EE-space figure-8 A=0.15 T=6 s centered at
EE(q0)=[0.5077, 0, 0.511], dt=0.01, N=64, SQP=1 (RTI), rho=0.01, cost EE=2
qd=1e-2 u=2e-6 N=50 qlim=0.01 mu=10, zero-control warm start. MPCGPU runs
`-DGATO_REG_PATTERN` + native eta-exit (the 2026-07-07 conditioning analysis:
cond(Pinv·S) ≈ 2e2, ~1.1 PCG iters/solve — see the old doc). Tracking = L2 at
the EE frame from logged joint configs, all three solvers.

| solver | median / solve | tracking L2 mean / max |
|---|---|---|
| **MPCGPU (GPU), GBD-PCG + GATO_REG_PATTERN, native exit** | **0.222 ms** | 0.0288 / 0.0635 |
| GATO (GPU, batch=1, closed loop) | 0.340 ms | 0.0295 / 0.0647 |
| BatchThneed (CPU, OSQP) | 3.11 ms | 0.0148 / 0.0736 |
| MPCGPU (GPU), QDLDL | (not re-timed this era) | 0.0294 / — (gate value) |

- MPCGPU-PCG remains the fastest single-solve engine (0.222 ms, p90 0.247,
  n=6010). QDLDL tracking re-verified by the gate (0.0294); its per-solve
  timing was last measured 2026-07-07 (0.384 ms, L7 era) and is expected
  unchanged in character — re-run `tools/time_persolve.sh 64 qdldl` if a
  quotable number is needed.
- BT's better mean tracking (0.0148) at 14× the latency is the usual
  accuracy/latency trade — 2 QP iterations per step vs the GPU solvers' RTI.

## GATO batch sweep (N=64) vs both baselines

Open-loop warm-started sweep (`sweep_batch_iiwa_fig8.py`, solver-internal
time, 400 solves/config minus warmup); BT closed-loop with num_threads=B
(24-core box); MPCGPU = B × its 0.222 ms single solve (no batch axis — one
cooperative grid owns the GPU).

| B | GATO ms total | GATO µs/traj | BT ms | GATO vs BT | GATO vs MPCGPU×B |
|---|---|---|---|---|---|
| 1 | 0.708* | 708 | 3.11 | 4.4× | 0.3× |
| 4 | 0.914 | 229 | 3.98 | 4.4× | 1.0× |
| 8 | 1.026 | 128 | 4.06 | 4.0× | 1.7× |
| 16 | 1.255 | 78 | 4.31 | 3.4× | 2.8× |
| 32 | 1.743 | 55 | 9.52 | 5.5× | 4.1× |
| 64 | 2.702 | 42 | 16.51 | 6.1× | 5.3× |
| 128 | 5.171 | 40 | 30.41 | 5.9× | 5.5× |
| 256 | 10.360 | 41 | — | — | — |
| 512 | 20.731 | 41 | — | — | — |

- **The July B=128 cliff is GONE** (was 10.8 ms / 84 µs-per-traj on the same
  part): GATO now amortizes monotonically to a flat **~40 µs/trajectory
  plateau holding from B=128 through B=512** (17.5× amortization from B=1).
  Attribution is the whole July wave (GLASS/GRiD bump + solver work), not a
  single change — the old sweet-spot advice "B≤64 at N=64" is obsolete.
- (*) Sweep-driver B=1 (0.708 ms) vs closed-loop 0.340 ms: the open-loop
  driver's warm-start/rho trajectory differs from the sim-fed loop. Use
  0.340 ms for the B=1 headline, the sweep for scaling shape (self-consistent
  across B).
- BT saturates its cores around B=16 (~250-270 µs/traj plateau in per-traj
  terms; per-solve latency keeps growing past that with oversubscription).

## GATO N × B grid (total batched solve ms, same run)

| N\B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.153 | 0.152 | 0.153 | 0.162 | 0.172 | 0.227 | 0.316 | 0.539 | 1.002 | 1.923 |
| 16 | 0.164 | 0.164 | 0.177 | 0.190 | 0.239 | 0.349 | 0.571 | 1.042 | 2.009 | 3.930 |
| 32 | 0.238 | 0.251 | 0.265 | 0.314 | 0.423 | 0.648 | 1.136 | 2.095 | 4.114 | 8.224 |
| 64 | 0.708 | 0.859 | 0.914 | 1.026 | 1.255 | 1.743 | 2.702 | 5.171 | 10.360 | 20.731 |
| 128 | 1.472 | 1.837 | 1.960 | 2.192 | 2.687 | 3.658 | 5.609 | 11.046 | 22.044 | 43.991 |

kHz-rate control: every N≤32 cell up to B=128, and N=64 to B≈8, sits under
1 ms (1 kHz); N=8..16 hold 1 kHz to B≈128-256.

## Reproduction

- `MPCGPU tools/run_3way_iiwa.sh` — builds gen_reference + validate_track with
  the fair flags, regenerates the EE-frame trajfiles, runs all three trackers
  (RESULT lines = the headline table's tracking columns).
- `MPCGPU tools/time_persolve.sh [N] [pcg|qdldl] [cycles] [csv]` — isolated
  per-solve timing (the 0.222 ms row).
- `GATO examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt
  --run-mpcgpu` — the sweep tables; CSVs land in GATO
  `examples/benchmarks/data/sweep_fig8_{gato,bt,mpcgpu}.csv` (fresh EE-era
  files; the L7-era CSVs are quarantined in
  `data/stale_L7_frame_pre20260730/`, see its README).
- Or all of it serially: `GATO examples/benchmarks/run_timing_night.sh`.

## Standing caveats (carried from the 07-06/07 analysis, still true)

- MPCGPU's historic full-Q+R regularization (the no-flag default) leaves
  cond(Pinv·S) ≈ 3e4 and a ~500× lying eta-exit; GATO_REG_PATTERN is the
  defensible operating point and the benchmark config. A better preconditioner
  remains the real future fix.
- Cadence differs by design: MPCGPU solves every 2 ms sim time; GATO/BT once
  per 10 ms step.
- All tracking is measured from logged joint configs at the shared EE frame
  (`GATO examples/benchmarks/iiwa_fig8_shared.py`, EE_FRAME="EE") — solver
  frame ≡ goal frame ≡ metric frame post-regen.
