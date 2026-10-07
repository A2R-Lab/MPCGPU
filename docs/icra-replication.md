# ICRA 2024 example replication

Goal: ship runnable versions of the principal examples from our MPCGPU paper,
including the iiwa five-goal pick-and-place task, with the updated solver stack.
Correctness comes first and needs no timing window. Performance collection waits
for an explicitly assigned quiet window and its own prepared manifest.

## Recovered protocol

The paper defers hyperparameters to its released code. The first code-and-experiments
commit (077252e, September 2023) and the April 2024 reorganization (c556b19, the
`main` branch) define the same protocol; they differ only in file layout and an
unused, default-off noise option. The table maps it onto the maintained task build
(`tools/build.py icra-pcg|icra-qdldl`, source `examples/icra_pick_place.cu`).

| Item | ICRA 2024 code | Maintained task build |
| --- | --- | --- |
| Robot model | iiwa14, pre-2026 inertials, end effector at the link-7 origin | Corrected model, named flange frame 4.0 cm further out |
| Gravity | 0 in the solver model and the simulator | 0 (`MPCGPU_GRAVITY=0`); the library default stays −9.81 |
| Reference | `examples/trajfiles/0_0_traj.csv`: 666 rows at 1/64 s, 10.4 s | Same file verbatim; EE targets recomputed with the current frame |
| Goals | Start pose, four goals, return to start | Same; segments begin at reference rows 2, 150, 296, 426 and 544 |
| Prediction model | Explicit Euler at 1/64 s | Same |
| Simulator | 0.2 ms Euler substeps, applies the previous solve's controls | Same |
| Control period | 2000 µs of simulated time per update (500 Hz) | Same |
| SQP stop | 2000 µs wall clock, at most 20 iterations when timing linear systems, or ρ above 10 | Correctness: 20 iterations, no wall clock, or ρ above 10 |
| Cost | ½‖p−p*‖² + ½·10⁻⁴‖q̇‖² + ½R‖u‖², R = 10⁻³ at N = 64, else 10⁻⁴; terminal knot without R | Same weights; Gauss-Newton JᵀJ position Hessian; terminal cost without the shared-memory alias |
| Regularization | ρ starts at 10⁻³, adapts by 1.2 within [10⁻³, 10], added to Q and R | Same |
| PCG | Absolute exit on η = rᵀP⁻¹r; iteration caps 173/167/167/118/67 for N = 32…512 | Same absolute test (`PCG_RES_TOL=0`) and caps |
| PCG tolerance | Per-horizon sweeps of five values | Middle entry of each sweep: 5·10⁻⁵ (N = 64), 10⁻⁴ (N ≥ 128); at N = 32 the next tighter sweep value, 2.5·10⁻⁶ (see below); `--tol` overrides |
| Warm-up | 100 solves at η = 10⁻¹¹, trajectory reset to the reference after each | Same (`WARMUP_RESET=1`) |
| Horizon shift | Refills the new last stage from the reference row at the current offset | Refills it from row offset + N − 1, the row the last stage represents (`REFERENCE_TAIL_FILL=1`) |
| Metric | L1 position error, sampled once per reference offset | L2 primary; L1 also reported |
| Trials | 100 repeats of one circuit with no randomness; runs differed only through the wall-clock SQP stop | Correctness builds are deterministic, so repeated trials must produce identical state hashes |
| Horizons | 32, 64, 128, 256, 512 | Same |

For N = 128 the middle sweep entry, 10⁻⁴, is the tolerance the paper names for
Figures 4 and 5. The paper does not state which tolerance produced the other
Figure 4 bars. The repository holds no run scripts, raw trial data or figure
scripts for the published results. Figure 6 depends on the wall-clock budget
and cannot be reproduced by a correctness build.

Differences kept on purpose: the corrected dynamics and EE frame, the Gauss-Newton
Hessian in place of the 2024 gradient outer product, the terminal-cost alias fix,
the corrected shift index and the L2 metric. Historical results computed with the
2024 code are therefore not directly comparable with current runs.

## Running the task

```bash
make icra BACKEND=pcg KNOT_POINTS=64     # builds, runs one trial, writes tmp/icra/pcg-N64
make icra BACKEND=qdldl KNOT_POINTS=128
```

`tools/icra_report.py` checks each run and writes `report.json` and `trajectory.svg`.
It requires finite data, identical repeated trials, all 666 offsets in reference
order, a closest approach under 5 cm to each of the five goals, the final goal held
within 1 cm, and mean L2 error under 10 cm. The fixture files are described in
[examples/icra/README.md](../examples/icra/README.md).

## Correctness evidence (September 28, 2026)

October 4 follow-up: after the graph-statistics repair, the full signed suite
passes both backends at N = 64 with repeated trials. Additional timer-disabled
single trials at N = 32, 128, 256 and 512 pass all task-quality checks for both
backends, including N = 32 PCG with the current tighter default tolerance.
These are simulated goal-tracking checks, not feasibility certificates: all
eight additional trials exceed URDF velocity limits (peak ratios 4.18–6.67),
and four also exceed position limits. The historical table below retains its
original date and settings. October 5 and 6 timing use repaired telemetry. After the
dependency update, eight further N32/128/256/512 trials pass with identical
pre-update state hashes; N64 repeats are covered by the full 100-test receipt.
Those correctness checks do not measure the new pins.

One trial per configuration, default tolerances, correctness builds on the RTX 5090 host.
Errors are end-effector position errors in meters; the approach column is the worst of
the five closest approaches. Early exits count control updates whose SQP loop ended
because ρ exceeded its maximum. These are correctness data, not timing results.

| N | Backend | Mean L2 | Mean L1 | Max L2 | Worst approach | Final | Early exits | Mean PCG iters | Joint excursion (rad) | Report |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | PCG | 0.0549 | 0.0826 | 0.202 | 0.114 | 0.0028 | 355 | 5.9 | 0.68 | fail: goal 2 at 0.114 |
| 32 | QDLDL | 0.0330 | 0.0483 | 0.239 | 0.031 | 0.0007 | 0 | — | 1.35 | pass |
| 64 | PCG | 0.0648 | 0.0982 | 0.273 | 0.036 | 0.0049 | 483 | 6.8 | 0.56 | pass |
| 64 | QDLDL | 0.0553 | 0.0815 | 0.325 | 0.031 | 0.0017 | 0 | — | 3.58 | pass |
| 128 | PCG | 0.0110 | 0.0168 | 0.058 | 0.023 | 0.0015 | 1815 | 6.0 | 0.00 | pass |
| 128 | QDLDL | 0.0178 | 0.0271 | 0.145 | 0.014 | 0.0002 | 0 | — | 0.49 | pass |
| 256 | PCG | 0.0116 | 0.0176 | 0.054 | 0.023 | 0.0010 | 1937 | 6.7 | 0.00 | pass |
| 256 | QDLDL | 0.0177 | 0.0270 | 0.149 | 0.014 | 0.0003 | 0 | — | 0.66 | pass |
| 512 | PCG | 0.0126 | 0.0193 | 0.063 | 0.022 | 0.0017 | 2110 | 6.0 | 0.00 | pass |
| 512 | QDLDL | 0.0176 | 0.0268 | 0.149 | 0.015 | 0.0003 | 0 | — | 0.00 | pass |

Repeated trials are bit-identical: the signed-suite gates run two trials at N = 64
for each backend and compare state hashes. The paper reports about 10 cm average L1
error; every configuration here stays below that.

## Timing results (October 6, 2026)

Collected on the release candidate — source `fa9f005` (receipt `22d6cec`), GRiD
`8dccbfa` and GLASS `9e57178` — on the RTX 5090 / Core Ultra 9 285K host, in one
assigned exclusive window with the ICRA, figure-eight and GATO legs back to back.
All 126 ICRA and 60 figure-eight repeats were revalidated against raw statistics,
tracking streams, plan and binary hashes. Artifacts:
`tmp/timing/release-20261005-9fa3Ue/{icra,fig8}` (local, not in Git). The
October 5 collection on the previous pins (`a684e78`, GRiD `0a14c0f`, GLASS
`8ce68a2`; `tmp/timing/audit-20261004-DRfuv2`) agrees with every table below
within repeat spread: Figure 4 means within 1 µs for PCG and 2% for QDLDL.

October 3 PCG iteration/exit telemetry was incomplete after graph replay;
October 5 and 6 use repaired, complete streams. Historical raw files are unchanged.

The iteration-rate grid uses fixed simulation periods and a soft wall-clock SQP
budget, not measured complete-controller deadlines. Every ICRA repeat has
internal-SQP overruns (65.0–100% of updates). PCG at N512/1 kHz has medians of
1158–1159 µs and p99 values of 1678–1685 µs. Pending GPU work must finish after
a time check; the timer also excludes the entry device wait and caller work.
QDLDL loses tracking at N512/1 kHz in all three repeats (maximum error 1.612 m).

The preceding October 3 experiment (`758bcd2`,
`tmp/timing/fixedcost-branch-20261003-115140`) first re-ran the
previous driver (`4317690`, `tmp/timing/fixedcost-main-20261003-115140`), which reproduced the
October 1 tables within 2 µs at every horizon. That experiment isolated the driver change
([speedup-attribution.md](speedup-attribution.md) section 4): the linear-system time no longer
includes two blocking statistics copies (−9 to −13 µs per solve) and each SQP step is shorter, so
more iterations fit a control period. October 6 retains that solver design and
has similar means, but is not a controlled A/B against October 3. The paper used an RTX 4090, an
i9-12900K and CUDA 12.1, and the model and EE frame have since been corrected. Compare ratios and
trends, not raw latency.

### Figure 4: average linear-system solve time at 500 Hz

| N | QDLDL (µs) | GBD-PCG (µs) | Speedup | Paper speedup | PCG mean L2 error (m) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 87 | 43 | 2.0× | 1.0× | 0.0401 |
| 64 | 161 | 43 | 3.8× | 1.5× | 0.0651 |
| 128 | 275 | 38 | 7.2× | 1.9× | 0.0107 |
| 256 | 539 | 47 | 11.5× | 3.6× | 0.0112 |
| 512 | 1101 | 52 | 21.0× | 3.3× | 0.0132 |

Across the three repeats, the per-repeat means spread by at most 8.5% of their mean for PCG and
6.8% for QDLDL. These are means of per-repeat means; 37–59% of PCG solves take
zero iterations from their warm start, making medians much lower.
Before the October 3 driver change the same workloads gave 55 / 54 / 49 / 57 / 65 µs
(1.6–16.5×) in the same window. The tables are printed by `tools/icra_tables.py` from the run directory. GBD-PCG time stays nearly flat with horizon here, so the speedups are much
larger than published. [The attribution](speedup-attribution.md) traces this to fewer PCG iterations
per solve, from controller fixes; GLASS and the new GPU contribute little.

### Figure 5: solve-time distribution at N = 128

QDLDL: median 273 µs, fastest 266 µs, 99.9th percentile 302 µs, slowest 1555 µs. Tails are
compared with the 99.9th percentile so that single outliers do not dominate.

| ε | Median (µs) | Mean (µs) | ≥10× faster than fastest QDLDL | Paper | Slowest / QDLDL p99.9 | ≥2× QDLDL p99.9 | Mean L2 error (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁻⁴ | 11 | 38 | 83% | 65% | 2.9× | 1.0% | 0.0107 |
| 5·10⁻⁵ | 11 | 46 | 79% | 52% | 4.1× | 1.2% | 0.0119 |
| 10⁻⁵ | 15 | 58 | 73% | 20% | 5.4× | 1.5% | 0.0140 |

The paper's ordering holds: tighter tolerances shift mass out of the fast mode, and PCG
keeps a slow mode near the QDLDL time.

### Figure 6: average SQP iterations per control step

| Solver | Rate | N = 32 | 64 | 128 | 256 | 512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QDLDL | 250 Hz | 18.8 (21) | 13.7 (14) | 9.2 (8) | 5.0 (4) | 2.1 (2) |
| QDLDL | 500 Hz | 9.0 (10) | 6.6 (6.5) | 4.7 (4) | 2.0 (2) | 1.0 (1) |
| QDLDL | 1000 Hz | 4.0 (4) | 3.0 (3) | 2.0 (1) | 1.0 (X) | lost (X) |
| GBD-PCG | 250 Hz | 24.2 (22.2) | 23.9 (19.7) | 19.9 (15.4) | 17.0 (5.2) | 10.8 (4.4) |
| GBD-PCG | 500 Hz | 12.5 (10.3) | 12.5 (10.6) | 10.9 (8) | 9.3 (4.6) | 5.3 (3) |
| GBD-PCG | 1000 Hz | 6.2 (4.9) | 6.3 (5.2) | 5.8 (3.7) | 4.9 (2.4) | 2.7 (1.7) |

Paper values are in parentheses. "lost" marks QDLDL at N = 512 and 1 kHz:
all three repeats lose tracking, and the harness records the rate as not met.
QDLDL averages about one iteration at N = 256 and 1 kHz and at N = 512 and
500 Hz. Tracking at those settings does not establish that the deadline was met.
GBD-PCG completes more iterations than in the paper at every horizon and rate (before the
October 3 driver change it did so at N ≥ 128; the shorter step adds about one iteration per
period at 500 Hz).

### Figure-eight workspace A/B (same source, not the paper task)

Internal SQP time per control update, medians of three repeats, from
`tmp/timing/release-20261005-9fa3Ue/fig8` (October 6 collection). Reused
workspaces keep allocations, streams, cuBLAS handles and — since October 3 — the page-locked
result arena and the captured launch graphs; fresh ones rebuild everything per solve,
approximating the earlier code (a per-call workspace does not page-lock or capture graphs).
Tracking is identical between the two modes. The previous driver measured 184 / 178 / 194 / 215 /
346 µs reused and 236 / 231 / 247 / 282 / 430 µs fresh for PCG in the same kind of window.

| N | PCG reused (µs) | PCG fresh (µs) | QDLDL reused (µs) | QDLDL fresh (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 139 | 237 | 221 | 327 |
| 64 | 134 | 233 | 279 | 409 |
| 128 | 155 | 241 | 409 | 594 |
| 256 | 189 | 276 | 705 | 987 |
| 512 | 312 | 421 | 1386 | 1849 |

## Findings

- **Gravity was the missing protocol input.** With −9.81 gravity and the paper's
  weights, both backends lose the circuit within 0.4 s (mean L2 error 0.34–0.40 m)
  because the control penalty pulls torques toward zero. The 2024 code ran in zero
  gravity. The task build restores that; the library default is unchanged.
- **The paper protocol does not respect joint limits.** Its cost has no joint-position,
  velocity or torque terms. The reference itself moves joints up to 7.5 times their
  URDF velocity limits, and the closed loop drives joint 3 past its position limit in
  the cost's redundant directions. The report records these excursions without
  failing, because the replicated protocol never enforced them.
- **PCG tolerance matters at short horizons.** At N = 32 with the sweep's middle value,
  5·10⁻⁶, PCG passes within 11 cm of the second goal and fails the 5 cm visit check. The
  tighter values from the paper's own N = 32 sweep, 2.5·10⁻⁶ and 10⁻⁶, bring every goal
  within 5 cm, so 2.5·10⁻⁶ is the N = 32 default since October 1, 2026, and the tables above
  use it: against the earlier 5·10⁻⁶ run (October 1 driver), N = 32 PCG needs 7.3 instead of 6.4
  iterations per solve (54 instead of 50 µs then, 1.6× instead of 1.8× over QDLDL) and its mean tracking error
  falls from 4.9 to 4.1 cm. This matches the paper's observation that looser
  tolerances trade tracking for speed.
- **Inexact PCG steps end some SQP loops early.** The line search fails until ρ
  exceeds its maximum on 24 to 1109 of 5204 control updates; QDLDL has none.
  Even so, PCG tracks more closely than QDLDL at N ≥ 128. The cause of that
  difference is not yet isolated.
- **N = 64 tracks worse than N = 32** because the paper used a ten times larger
  control weight at N = 64.

## Checklist

- [x] Inventory the original example entry points, trajectories and settings.
- [x] Record model, frame, goals, initial conditions, timestep, integrator, costs,
  regularization, warm start, solver limits, tolerances and control-update policy.
- [x] Recover the trial-generation procedure. There were no seeds or perturbations.
- [x] Keep corrected dynamics and safety fixes; document every difference above.
- [x] Shared PCG/QDLDL task build with a one-command example and machine-readable output.
- [x] Deterministic single-trial and repeated-trial checks; signed-suite gates at N = 64.
- [x] Trajectory plot from each run (`trajectory.svg`).
- [x] Signed receipt `513a9a4` (86 passed, no skips) and a fresh-clone quickstart:
  public HTTPS submodules, a new virtual environment, host tests, both demos and
  `make icra` for both backends reproduced the numbers above exactly (repeated October 1
  for the main integration candidate).
- [x] Paper-task timing manifest: `tools/timing.py prepare --task icra` builds the
  Figure 4, 5 and 6 workloads (see the timing section of [development](development.md)).
- [x] Paper-task timing re-collected on October 5 with repaired telemetry (superseded below).
- [x] Re-measured on the release dependency pins on October 6 (tables above).
- [x] Paper-task timing collection on October 1 (the September 30 collection
  preceded the N = 32 tolerance change), labeled by host.
- [x] Attribute the larger-than-published PCG speedups ([attribution](speedup-attribution.md), October 1).
- [x] Website shows the current results with hardware and date; the paper holds the
  historical results.

Exit criterion: the principal paper tasks are runnable and checked on the current
stack; qualitative behavior and measured comparisons are explained and
reproducible. Correctness and the first timing collection are complete for the
pick-and-place circuit, and the change from the published speedups is attributed.
