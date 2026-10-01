# Why the ICRA results moved (attribution, October 1, 2026)

For review before the main merge. The September 30 timing on the RTX 5090 gave GBD-PCG
average linear-system times of 50–64 µs at every horizon, against 75–360 µs in the paper.
QDLDL on the CPU stayed within about 10% of the paper, so the speedup over QDLDL grew from
1.0–3.6× to 1.8–17×. This document splits that change into its causes.

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

PENDING_KERNEL

## 3. Hardware and toolchain

PENDING_HARDWARE

## Summary

PENDING_SUMMARY

## Reproduce

```bash
bash tools/attribution/prepare.sh                     # builds and captures; measures nothing
MPCGPU_QUIET_WINDOW=1 bash tools/attribution/run.sh   # exclusive quiet window only
```

The variant walk in section 1 patched only a scratch copy of the paper-era code; the patches are
listed in the table and are not part of the maintained tree.
