# MPCGPU modernization onto GRiD + GLASS (Part 2 of the A2R-Lab unification)

This documents the rewire of MPCGPU and GBD-PCG (a submodule at the time; folded in-tree 2026-08) to defer dynamics to GRiD (a regenerated
`grid.cuh`) and in-block linear algebra to GLASS, mirroring GATO. Goal: future GRiD/GLASS bumps are a
`regen` + pin bump instead of constant sm_120/smem/GLASS-API hand-patches.

## What changed

**GBD-PCG (`modernize-glass`):**
- `bdmv` (per-block block-tridiagonal matvec) → `glass::gemv` (col-major `ROW_MAJOR=false`, matches the
  existing `[L|D|R]` strip storage as-is) with the absent L/R boundary strips zero-padded so the three
  boundary cases collapse to one full-width matvec. Cooperative `grid.sync()` structure untouched.
- `dot`/`reduce` ported to the current GLASS API (`dot_lowmem`, copy-then-in-place `reduce`).
- `gpuassert.cuh` guarded with `#ifndef gpuErrchk` (ODR vs grid.cuh, which defines it unconditionally).
- New `examples/test_pcg_spd.cu`: SPD block-tridiagonal residual gate (the shipped `pcg_solve.cu` NaNs on
  an uninitialized preconditioner).

**MPCGPU (`modernize-grid-glass`):**
- Added the `GRiD` submodule + `tools/regen_grid.py` + vendored `tools/iiwa14.urdf`. Regenerated
  `include/dynamics/iiwa/grid.cuh` (byte-identical to GATO's iiwa14 grid.cuh); deleted the dead vendored
  grid headers.
- Rewrote `iiwa_eepos_plant.cuh` to GRiD's current `*_inner` API (`s_topology_helpers`, `d_workspace`/
  `d_f_ext`, gravity as the trailing arg, `minv_inner`, EE pose via `end_effector_pose[_gradient]_inner`).
- Migrated the hand-rolled `include/utils/matrix.cuh` primitives to `glass::` across `dz.cuh`,
  `pcg/linsys_setup.cuh`, `qdldl/linsys_setup.cuh` (validated bit-exact; see below).
- Cost fixes in `iiwa_eepos_plant.cuh`: EE Gauss-Newton Hessian corrected to `JᵀJ` (was the rank-1
  gradient outer product); added `Q_COST` joint-posture regularization (default 0).
- `Makefile`: `ARCH ?= sm_120` + `submodules`/`regen`/`test_fd_parity` targets + `-DNDEBUG`.

## The corrected-dynamics finding

The *old vendored* grid.cuh had a ~2×-wrong mass matrix; the regenerated grid is **pinocchio-exact**
(verified: `forwardDynamics` qdd == `pin.aba` to 6 digits; `df_du` and `end_effector_pose_gradient` FD-exact
to 1e-9/1e-11). The correct iiwa is stiff — last link inertia ≈ 0.003 → `Minv[6,6] ≈ 392`.

Consequence: the modernized MPC **diverges** on the shipped `examples/trajfiles/`. This is NOT a bug:
- Per-step SQP converges (~6 iters, never hits the cap); QDLDL (exact solve) also diverges → not a solver
  issue.
- The cost is EE-position-only; joint 7's EE gradient is identically zero, so it sits in the cost nullspace
  and, with `Minv≈392`, is an unstable free integrator on a reference generated for the wrong robot.
- Tested exhaustively: no `R/QD/Q_COST` combination, `JᵀJ` Hessian, or semi-implicit-Euler prediction
  recovers tracking. The original "tracked" only because its wrong heavier dynamics were forgiving.

**Tracking quality is therefore not a valid modernization gate.** Reproducing the benchmark needs
regenerated trajectories for the correct robot (no in-repo trajopt pipeline) or a controller redesign —
tracked as a follow-up. The modernization itself is correctness-complete.

## Validation

- Dynamics: `make test_fd_parity` (adapter vs `grid::forward_dynamics_device`, max err 0); FD self-tests on
  `df_du` and the EE pose gradient; `pin.aba` cross-check.
- `grid.cuh` byte-diff vs GATO.
- GBD-PCG: SPD residual gate + compute-sanitizer clean.
- matrix.cuh → glass migration: every primitive (`gemv` transpose/col-major, `axpby`, `addI`,
  `loadIdentity` incl. the multi-matrix split, `invertMatrix` incl. fused 2/3-matrix) checked **bit-exact**
  against the hand-rolled versions on random inputs; PCG end-to-end tracking number bit-identical pre/post
  migration; compute-sanitizer clean. (The QDLDL diverging closed loop is inherently non-deterministic and
  is not a usable regression signal.)
