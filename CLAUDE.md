# MPCGPU — agent guide

Real-time nonlinear MPC on the GPU (paper: arXiv:2309.08079). A single cooperative GPU-wide
preconditioned-conjugate-gradient (PCG) SQP solver, iiwa14-only, Makefile build.

## Architecture (post-modernization)

MPCGPU is a thin solver that **defers dynamics to GRiD and in-block linear algebra to GLASS**, mirroring
GATO. Future GRiD/GLASS bumps are a `regen` + pin bump, not hand-patching.

- **Dynamics → GRiD.** `include/dynamics/iiwa/grid.cuh` is generated from the pinned `GRiD` submodule
  (`tools/regen_grid.py`, single iiwa14 entry, profile="all"). It is **byte-identical to GATO's**
  `iiwa14/grid.cuh` (the cross-check that catches URDF divergence). The adapter
  `include/dynamics/iiwa/iiwa_eepos_plant.cuh` wraps GRiD's `*_inner` kernels (forward/inverse dynamics,
  fd/id gradient, minv, end_effector_pose[_gradient]). No dynamics are hand-rolled.
- **In-block linalg → GLASS.** The block-scoped primitives (`gemv`, `axpby`, `addI`, `loadIdentity`,
  `invertMatrix`, `gemm`, `copy`, `reduce`, `dot`) come from the top-level `glass::` namespace. The old
  hand-rolled `include/utils/matrix.cuh` primitives were migrated to `glass::` (bit-exact validated);
  only the host-side `write_device_matrix_to_file` debug dumper remains there.
- **Cooperative PCG → GBD-PCG.** `GBD-PCG/` is the **cooperative, grid-wide** block-tridiagonal PCG
  (`cudaLaunchCooperativeKernel`, one block per knot, `grid.sync()`). It is the cooperative analog of
  GLASS's **single-block** `glass::pcg`/`glass::bdmv`. Its per-block matvec (`bdmv`) defers to
  `glass::gemv` (col-major, zero-padded boundaries); its in-block ops are `glass::copy/dot/reduce`.

## Submodules / pins

`GRiD` (dynamics codegen), `GLASS` (linalg), `GBD-PCG` (cooperative PCG; **has its own nested `GLASS`**).
Bump protocol: move the pin, `git submodule update --init --recursive`, rebuild, revalidate. When bumping
GLASS, bump it in **both** `GLASS/` and `GBD-PCG/GLASS/` and keep them equal.

## Build / run

```
make submodules          # git submodule update --init --recursive
make build_qdldl
make examples            # ARCH?=sm_120 (override: make ARCH=sm_86 examples)
mkdir -p tmp/results
LD_LIBRARY_PATH=$PWD/qdldl/build/out ./examples/pcg.exe     # run from repo root (trajfile paths)
```
Parameters in `include/common/settings.cuh` are all `#ifndef`-guarded → override with `-D` (see Makefile).

## Validation gates (correctness-only; may run alongside other GPU work)

- `make test_fd_parity` — adapter `forwardDynamics` vs `grid::forward_dynamics_device` (expect max err 0).
- `grid.cuh` byte-diff vs GATO's iiwa14 grid.cuh (URDF-divergence gate).
- `GBD-PCG/examples/test_spd.exe` — SPD block-tridiagonal residual gate for the cooperative solver.
- Primitive migrations were validated **bit-exact** vs the hand-rolled originals.

## ⚠ Corrected dynamics + the tracking benchmark

The regenerated grid is **pinocchio-exact**; the *old vendored* grid had a ~2×-wrong mass matrix. The
correct iiwa is **stiff** (last-joint inertia ≈ 0.003 → `Minv[6,6] ≈ 392`). The shipped
`examples/trajfiles/` (warm-start + reference) were generated for the *wrong* robot, so the EE-position-only
MPC is **closed-loop unstable** on them with the corrected dynamics (verified: per-step SQP converges, all
gradients FD-exact, yet the closed loop diverges; not fixable by cost weights). **Tracking quality is NOT a
valid gate for the modernization.** Reproducing the tracking benchmark needs regenerated trajectories for the
correct robot (no in-repo trajopt) or a controller redesign — logged as a follow-up.

Cost notes (`iiwa_eepos_plant.cuh`): the EE-position cost Hessian is the true Gauss-Newton `JᵀJ` (the legacy
code used the rank-1 gradient outer product `(Jᵀe)(Jᵀe)ᵀ`, fixed). `Q_COST` (default 0) adds joint-posture
regularization toward `q_nom=0` to anchor the EE-nullspace joints. See `docs/modernization.md`.

## Conventions

Short single-line commits. Hold pushes / upstream PRs until explicitly cleared. Never time perf under GPU/CPU
contention (correctness runs may overlap; only timing needs isolation).
