> Historical checkpoint, not current instructions or candidate performance evidence.
> See [current documentation](README.md) and [implementation status](implementation-status-2026-09-28.md).

# N-sweep: per-solve cost vs horizon length (2026-08-03)

Modern retest of the MPCGPU paper claim — *"kHz control rates with trajectories as
long as 512 knot points"* (arXiv:2309.08079) — on the 2026 stack (GRiD dynamics,
GLASS linalg, in-tree GBD-PCG, Drake-family iiwa14).

**Verdict: the claim holds for the cooperative PCG solver with 2.4× headroom
(416 µs @ N=512 ⇒ 2.4 kHz) and fails for the direct QDLDL solve (1.84 ms ⇒ 543 Hz).**
The interesting result is not the endpoint but the *shape*: cooperative PCG is
horizon-**independent** to N≈128 and grows at 0.57 µs/knot thereafter, while QDLDL
is linear at ≈3.4 µs/knot from the start.

## Provenance

- HEAD `c3c2df2` (post Drake-family iiwa14 swap), `tools/time_persolve.sh`, B=1.
- Fair config: `-DGATO_REG_PATTERN`, native eta-exit, SQP=1, PCG cap 200, rel tol
  1e-4, RHO_INIT 0.01 — identical to the 3-way benchmark
  (`docs/benchmark_3way_2026-08-01.md`).
- 1202 sim offsets → 6010 solves per cycle; 3 cycles per cell; reported value is
  the median of the 3 run-medians.
- Quiet-box preflight recorded: util 0%, 0 compute apps, loadavg 1.16 (a decaying
  average from prior activity, not concurrent work — the 3 cycles per cell agree
  to <0.5%, so no cell is load-contaminated).
- Raw: `tmp/nsweep_logs/20260803_024006/` (gitignored — SUMMARY.txt + per-leg logs
  + `nsweep_persolve.csv`; ⚠ the CSV carries no `linsys` column, rows are pcg 8..512
  then qdldl 8..512 in loop order).

## Results

| N | PCG µs | PCG Hz | QDLDL µs | QDLDL Hz | QDLDL/PCG | PCG µs/knot | QDLDL µs/knot |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 231.7 | 4316 | 239.0 | 4184 | 1.03× | 28.96 | 29.88 |
| 16 | 229.2 | 4363 | 261.7 | 3821 | 1.14× | 14.33 | 16.36 |
| 32 | 228.6 | 4374 | 307.0 | 3257 | 1.34× | 7.14 | 9.59 |
| 64 | 221.8 | 4509 | 389.0 | 2571 | 1.75× | 3.47 | 6.08 |
| 128 | 235.5 | 4246 | 578.5 | 1729 | 2.46× | 1.84 | 4.52 |
| 256 | 271.0 | 3690 | 973.9 | 1027 | 3.59× | 1.06 | 3.80 |
| 512 | 416.2 | 2403 | 1842.8 | 543 | 4.43× | 0.81 | 3.60 |

Marginal cost of each *added* knot (finite difference across each doubling):

| doubling | PCG µs/knot | QDLDL µs/knot |
|---|---:|---:|
| 8→16 | −0.31 | 2.84 |
| 16→32 | −0.04 | 2.83 |
| 32→64 | −0.21 | 2.56 |
| 64→128 | +0.21 | 2.96 |
| 128→256 | +0.28 | 3.09 |
| 256→512 | +0.57 | 3.39 |

## Reading

1. **PCG is flat to N=128.** N=8…128 spans 221.8–235.5 µs — a ±3% band, narrower
   than the p90/median jitter of the cooperative launch itself (1.13–1.20 vs QDLDL's
   1.02–1.04). Per-solve cost in that regime is a ~225 µs *floor* — cooperative
   launch + grid.sync barriers + KKT setup latency — not linear-algebra work. At the
   fair config the solver averages ~1.1 PCG iterations/solve, so there is very little
   linear algebra to be horizon-dependent about.
2. **The growth past N=128 is not occupancy loss.** This GPU's co-resident cap is
   ~850 blocks at the default 64-thread block, so all 512 knot-blocks are still
   co-resident at the top of the sweep. The 1.9× rise off the floor is consistent
   with grid-wide reduction/sync *depth* (the block-tridiagonal sweep and the
   dot-product reductions across N blocks) rather than blocks queueing for SMs.
3. **QDLDL is linear from the first doubling**, at 2.6–3.4 µs/knot with a mild
   upward drift (cache-resident → not). Amortized cost asymptotes to 3.60 µs/knot.
4. **kHz budget.** At a 1 ms budget: PCG clears N=512 with 2.4× headroom; QDLDL
   crosses 1 ms at **N ≈ 264** (interpolating 973.9 µs @256 → 1842.8 µs @512).
   Extrapolating PCG's 0.57 µs/knot marginal, the 1 ms budget would not bind until
   N ≈ 1500 — meaning **for cooperative PCG the binding constraint at kHz rates is
   block co-residency (~850 knots on this GPU), not the time budget.** That is an
   extrapolation past measured data; the honest measured statement is the 2.4×
   headroom at 512.
5. **Timing is model-swap-invariant, confirmed empirically.** N=64 PCG here is
   221.8 µs against the pre-swap 3-way doc's 0.222 ms — the Drake-family inertial
   swap moved tracking slightly and timing not at all, as predicted when the sweep
   was launched without waiting for the swap.

## Tracking quality (the non-timing finding)

Mean EE tracking error on the fig8 reference, same runs:

| N | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCG | 0.02407 | 0.02306 | 0.02727 | 0.02865 | 0.03017 | 0.03028 | 0.03034 |
| QDLDL | 0.02068 | 0.02450 | 0.02753 | 0.02939 | 0.03021 | 0.03033 | 0.03032 |

- **Longer horizons track *worse* here, monotonically past N=16.** At SQP=1 the
  solver takes one Newton step per control tick; lengthening the horizon spreads
  that single step over more knots and makes the applied first control less
  aggressive on a reference that is fully known and perfectly trackable. Long
  horizons are *affordable* on this hardware — they are not *useful* on this task.
- **PCG converges to the exact solve as N grows**: the PCG/QDLDL tracking gap is
  16% at N=8 and 0.07% at N=512. Inexactness of the iterative solve matters only
  where the horizon is short enough for the first control to be sensitive to it.
- N=64 reproduces the committed gate truths exactly (PCG 0.028647, QDLDL 0.029387),
  which cross-checks that the sweep ran the gate configuration.

## Consequences

- The paper's headline claim survives a full stack modernization and a robot-model
  correction; quote **416 µs @ N=512 (2.4 kHz)** for PCG and note QDLDL's 543 Hz as
  the direct-solve contrast.
- N=512 is a *capability* datapoint, not a recommended operating point: N=32–64 is
  both cheaper and better-tracking on this task.
- Any future N-sweep should record the `linsys` in the CSV (currently implied by
  row order only).
