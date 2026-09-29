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
| PCG tolerance | Per-horizon sweeps of five values | Middle entry of each sweep: 5·10⁻⁶ (N = 32), 5·10⁻⁵ (N = 64), 10⁻⁴ (N ≥ 128); `--tol` overrides |
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
- **PCG tolerance matters at short horizons.** At N = 32 with the default 5·10⁻⁶,
  PCG passes within 11 cm of the second goal. The tighter values from the paper's
  own N = 32 sweep, 2.5·10⁻⁶ and 10⁻⁶, bring every goal within 5 cm. This matches
  the paper's observation that looser tolerances trade tracking for speed.
- **Inexact PCG steps end some SQP loops early.** The line search fails until ρ
  exceeds its maximum on 355 to 2110 of about 5200 control updates; QDLDL has none.
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
- [ ] Refresh the full signed receipt with the new gates, then fresh-clone verification.
- [ ] Optional variant with joint-limit terms, if physically valid motion is wanted.
- [ ] Paper-task timing: prepare immutable binaries and a manifest, then collect
  linear-system time, controller time and task quality for both backends in an
  assigned quiet window. Report them separately and label the RTX 5090 /
  Core Ultra 9 285K / CUDA 13.2 host against the paper's RTX 4090 / i9-12900K /
  CUDA 12.1. Exact old latency values are not the replication target.
- [ ] Add current results beside the published ones on the website, labeled by
  hardware and method, only from matching experiment data.

Exit criterion: the principal paper tasks are runnable and checked on the current
stack; qualitative behavior and measured comparisons are explained and
reproducible. Correctness is complete for the pick-and-place circuit; the
performance comparison is not.
