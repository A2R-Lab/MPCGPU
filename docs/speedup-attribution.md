# Why the ICRA results moved (attribution, October 1–3, 2026)

The October 1, 2026 timing on the RTX 5090 gave GBD-PCG average linear-system times of
49–63 µs at every horizon, against 75–360 µs in the paper. QDLDL on the CPU stayed within
about 15% of the paper, so the speedup over QDLDL grew from 1.0–3.6× to 1.6–17×. This
document splits that change into its causes. Section 4 adds the October 3 driver change,
after which the same workloads give 38–53 µs and 2.1–20× ([current tables](icra-replication.md)).

Average PCG solve time is iterations per solve times time per iteration, plus a fixed launch
and copy-back cost. Each factor was measured separately.

## 1. Iterations: the controller's linear systems got easier

Every kernel version takes exactly the same number of iterations on the same system. On one
captured N = 128 system from each pipeline, at the paper's absolute tolerance 10⁻⁴:

| Captured system | Paper kernel | Same kernel on GLASS | Current kernel |
| --- | ---: | ---: | ---: |
| Paper-era pipeline | 205 | 205 | 205 |
| Current pipeline | 18 | 18 | 18 |

So the iteration drop comes from the systems the controller produces, not from GBD-PCG or GLASS.
The paper-era system is symmetric and definite, so it was well formed. It has a 340× larger
right-hand side (182 versus 0.54) and a 2.9× larger preconditioned condition number
(2.9·10⁵ versus 1.0·10⁵). With an absolute exit test, a larger right-hand side needs many more
iterations to reach the same tolerance.

Which changes produced the easier systems: the paper-era code was rebuilt with each fix toggled
and run over the full circuit at N = 128, ε = 10⁻⁴ (correctness builds, 20 SQP iterations per step):

| Paper-era code variant | Mean PCG iterations | Solves hitting the cap | Mean L1 EE error |
| --- | ---: | ---: | ---: |
| As published | 18.8 | 5.8% | 6.0 cm |
| Horizon-tail index fixed | 11.5 | 3.3% | 2.0 cm |
| Gauss-Newton position Hessian | 5.2 | 0.7% | 3.9 cm |
| Both | 3.9 | 0.5% | 2.1 cm |
| Current code (corrected model and frame, terminal fix) | 6.0 | 0.5% | 1.7 cm |

- **Gauss-Newton Hessian (largest effect).** The 2024 cost used the outer product of the cost
  gradient as the position Hessian, a rank-1 matrix that vanishes as tracking error shrinks.
  JᵀJ keeps the EE curvature, which conditions the Schur system and cuts iterations 3.6×.
- **Horizon-tail index.** On every shift, the 2024 code refilled the horizon's last stage from
  the reference row at the current time instead of the row the last stage represents. That
  injected a large dynamics defect at the horizon's end, inflating the right-hand side.
- The corrected dynamics model and EE frame add back about two iterations per solve.

## 2. Time per iteration: kernel and GLASS

Identical captured systems, solved for a fixed 25 and 100 iterations with exit tolerance 0, three
alternating repeats of 300 timed solves each (October 1, 2026, exclusive window). The per-iteration
cost is the slope between the two; the one-iteration solve is the fixed launch and copy-back cost.

| N | Paper kernel (µs/iter) | On GLASS (µs/iter) | Current (µs/iter) | GLASS change | One-iteration solve, paper / current (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4.91 | 4.72 | 4.62 | -3.9% | 23.0 / 22.5 |
| 64 | 4.97 | 4.86 | 4.86 | -2.3% | 22.8 / 22.9 |
| 128 | 5.22 | 4.97 | 4.94 | -4.6% | 22.7 / 22.6 |
| 256 | 6.21 | 6.06 | 6.06 | -2.4% | 24.6 / 24.6 |
| 512 | 8.20 | 8.14 | 8.16 | -0.7% | 34.1 / 34.6 |

Moving GBD-PCG onto GLASS (commit `bc60729`, its only change) cut the time per iteration by 0.7–4.6%,
most at short and medium horizons. Later GBD-PCG changes, the relative tolerance, converged-start
guard and safety checks, left it unchanged. The fixed cost of about 23 µs at N ≤ 256 is the same in
every version, and it was most of the October 1 49–63 µs average because current solves need few iterations.

![Time per PCG iteration by kernel version](attribution-per-iteration.png)

## 3. Hardware and toolchain

The published code was rebuilt unchanged for this GPU and run exactly as in the paper: 500 Hz,
2000 µs wall-clock SQP budget, linear-system timers, its middle tolerances, three repeats. Its
speedups over QDLDL are 1.2–2.9×, the same range as the published 1.0–3.6× and far from the current
code's 1.6–17×, so the new hardware does not explain the new results. Point by point it is close to the
published GBD-PCG times at N = 32, 128 and 512, faster at 64 and slower at 256. Its PCG time varies
strongly between runs: at N = 128 the three repeats gave 333, 152 and 138 µs, against about 150 µs read
from the published Figure 4. QDLDL on the CPU matches the published bars within about 15%.

| N | Paper code PCG (µs, median [range]) | Paper code iterations | Current PCG (µs) | Current iterations | Paper code QDLDL (µs) | Current QDLDL (µs) | Paper code speedup | Current speedup | Published speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 71 [71–72] | 10.9 | 54 | 7.3 | 86 | 89 | 1.2× | 1.6× | 1.0× |
| 64 | 64 [62–65] | 9.2 | 54 | 7.0 | 144 | 162 | 2.2× | 3.0× | 1.5× |
| 128 | 152 [138–333] | 25.8 | 49 | 5.6 | 276 | 277 | 1.8× | 5.6× | 1.9× |
| 256 | 283 [253–305] | 42.3 | 57 | 6.2 | 543 | 535 | 1.9× | 9.4× | 3.6× |
| 512 | 367 [362–391] | 41.8 | 63 | 5.1 | 1078 | 1074 | 2.9× | 17.1× | 3.3× |

Current-code columns are means over the three repeats of the October 1 evening run (N = 32 at its
2.5·10⁻⁶ default), the statistic the plots and the replication notes use; paper-code columns are medians with the range because of the N = 128 outlier.

![Published versus current code on this host](attribution-paper-vs-current.png)

## 4. Fixed cost (October 3, 2026): host synchronization in the SQP step

Section 2's one-iteration solve put the fixed launch and copy-back cost at about 23 µs per linear
system. Profiling a whole SQP step (figure-eight task, N = 64, `nsys`) showed where the rest of a
control update went: the KKT formation (37 µs) and Schur assembly (43 µs) dominate, PCG takes 6–19 µs,
dz 4 µs and the eight line-search merits 19 µs running concurrently — about 123 µs of kernels in a
178 µs step. The remaining 55 µs were host synchronization: two device syncs around the linear
system, two blocking copies of the PCG statistics, an implicit sync plus a blocking copy for the
merits, a redundant sync after setup, and a 17 µs initial-merit kernel run serially ahead of the KKT
formation. None of it was kernel launch cost: the host finished enqueueing a step in about 30 µs.

The SQP driver now runs the step on one stream with one host wait per step (after the merits, to
pick the step and the next rho), the initial merit on a side stream beside the KKT formation, the
line-search merits forked to eight streams and joined, the per-step segments captured once per
reused workspace as CUDA graphs, and the few words the host needs (merits, PCG iteration count and
exit flag) stored by the kernels straight into mapped page-locked host memory. Every STATE_HASH gate
and the ICRA circuits are bit-identical to the previous driver; the figure-eight tracking results are
identical to six digits in all twenty timing workloads.

Figure-eight plan (`tools/timing.py`, three repeats, median internal SQP time per control update,
reused workspace), previous driver (`main` 4317690, exclusive leg, October 3 00:16) versus the new
driver (758bcd2, exclusive leg, October 3 01:43):

| N | pcg before (µs) | pcg after (µs) | change | qdldl before (µs) | qdldl after (µs) | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 184.0 | 137.8 | -25% | 243.7 | 221.0 | -9% |
| 64 | 178.4 | 133.5 | -25% | 302.5 | 275.8 | -9% |
| 128 | 193.8 | 154.7 | -20% | 453.6 | 415.2 | -8% |
| 256 | 214.7 | 188.6 | -12% | 753.7 | 716.3 | -5% |
| 512 | 345.7 | 311.3 | -10% | 1447.3 | 1381.1 | -5% |

The gain is a fixed amount per step, so it matters most at short horizons where the kernels are
cheap. The per-call ("fresh") workspace workloads are unchanged within noise: a per-call workspace
does not page-lock memory (that alone costs about 0.4 ms) and launches the segments directly.

Two measurement notes. First, `linsys_times` (ICRA Figures 4/5) keeps the paper's definition, host
wall time between a device sync before and after the linear system, so `TIME_LINSYS` builds still pay
those two syncs per step; the figure-eight numbers above include them. Second, that linear-system
time itself drops by about 13 µs per solve (23.4 → 10.5 µs at N = 64 on the figure-eight task) because
the two blocking 4-byte copies of the PCG statistics that used to sit inside the timed region are gone.
Re-collecting the ICRA workloads in the same window (October 3, both drivers back to back,
[icra-replication.md](icra-replication.md)) showed exactly that: GBD-PCG 55 / 54 / 49 / 57 / 65 µs
→ 42 / 42 / 38 / 48 / 53 µs at N = 32…512 (−9 to −13 µs), speedups 1.6–16.5× → 2.1–20.4×, and about
one more SQP iteration per 500 Hz period; the kernels themselves are unchanged.

Collection: both legs ran from `tools/timing.py` plans prepared from their own commits, each in a
quiet window with no other process on the GPU (`a2rlab-timing-chain/runs/fixedcost-ab-20261003-001557`
and `-014302`), three repeats per workload; all sixty repeats of each leg produced a verdict.

## Summary

- **Iterations explain the change.** On the same machine and task, the current code needs 5–7 PCG
  iterations per solve at every horizon, the published code 9–42, growing with horizon. That makes
  today's GBD-PCG time nearly flat in N, while QDLDL still grows linearly, so the ratio grows with N.
- **The iteration drop comes from the controller, not the solver.** All kernel versions converge in the
  same number of iterations on a given system. The Gauss-Newton position Hessian conditions the Schur
  systems; the corrected horizon-tail index removes an injected defect that inflated their right-hand
  side. Both also improve tracking.
- **GLASS made each iteration 0.7–4.6% cheaper.** That is a real but small part of the change.
- **The hardware change is not the cause.** The published code on the RTX 5090 gives speedups in the
  published range, 1.2–2.9×.
- **Host synchronization was the fixed cost** (section 4, October 3). Removing it shortens a control
  update by 25% at N = 32–64 and 10% at N = 512 with bit-identical results; the kernels are unchanged.

These are internal linear-system times on one host. The paper's reported speedups remain correct for
the published code; the current speedups describe the current code.

## Reproduce

```bash
bash tools/attribution/prepare.sh                     # builds and captures; measures nothing
MPCGPU_QUIET_WINDOW=1 bash tools/attribution/run.sh   # exclusive quiet window only
```

The variant walk in section 1 applies `tools/attribution/paper-variants.patch` to the paper-era code
extracted by `prepare.sh` (`tmp/attribution/paper-code`), building with `-DFIX_TAIL_INDEX`,
`-DGN_HESSIAN`, `-DONLY_TOL_INDEX=2` and `-DDUMP_AT=<solve>` as needed. The patch is a measurement aid
for the historical code only; it is not part of the maintained solver.
