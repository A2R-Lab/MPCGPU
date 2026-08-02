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
- **In-block linalg → GLASS.** The block-scoped primitives (`gemv`, `axpby`, `add_identity`,
  `set_identity`, `inv`, `gemm`, `copy`, `reduce`, `dot`) come from the top-level `glass::`
  namespace (GLASS naming r2, pin ≥5caa6d0: `addI→add_identity`, `loadIdentity→set_identity`,
  `invertMatrix→inv`, `addI_partial→add_identity_partial`). The old
  hand-rolled `include/utils/matrix.cuh` primitives were migrated to `glass::` (bit-exact validated);
  only the host-side `write_device_matrix_to_file` debug dumper remains there.
- **Cooperative PCG → GBD-PCG.** `GBD-PCG/` is the **cooperative, grid-wide** block-tridiagonal PCG
  (`cudaLaunchCooperativeKernel`, one block per knot, `grid.sync()`). It is the cooperative analog of
  GLASS's **single-block** `glass::pcg`/`glass::bdmv`. Its per-block matvec (`bdmv`) defers to
  `glass::gemv` (col-major, zero-padded boundaries); its in-block ops are `glass::copy/dot/reduce`.
  **In-tree since 2026-08** (folded from the retired standalone A2R-Lab/GBD-PCG repo, full history
  preserved as a subtree merge); it compiles against the top-level `GLASS/` submodule.

## Submodules / pins

`GRiD` (dynamics codegen), `GLASS` (linalg), `qdldl`. GBD-PCG is **in-tree** (not a submodule) —
there is exactly ONE GLASS pin for the whole repo. Bump protocol: move the pin,
`git submodule update --init --recursive`, rebuild, revalidate (GBD-PCG picks the new GLASS up
automatically via `-I../GLASS` / `-IGLASS`).

## Build / run

```
make submodules          # git submodule update --init --recursive
make build_qdldl
make examples            # ARCH?=sm_120 (override: make ARCH=sm_86 examples)
mkdir -p tmp/results
LD_LIBRARY_PATH=$PWD/qdldl/build/out ./examples/pcg.exe     # run from repo root (trajfile paths)
```
Parameters in `include/common/settings.cuh` are all `#ifndef`-guarded → override with `-D` (see Makefile).

## Validation gates (correctness-only)

**GPU CI = pytest-gpu-proof (mirrors GATO, added 2026-08-01):** `test/test_gates.py` wraps every
gate below as a pytest suite; `./test/run_gpu_proof.sh` runs it on the GPU box (bootstraps
`.venv`, builds qdldl if needed, requires a clean tree + the sibling `../GATO` checkout) and
signs `gpu-proof.json`; commit the receipt and the CPU-only `verify-gpu-proof` workflow checks
it on every push. Config in `pyproject.toml [tool.gpu_proof]` + `test/gpu-proof-policy.yaml`.
Changes under `include/`, `tools/`, `examples/`, `test/`, or `Makefile` change the fingerprint —
regenerate the receipt with such a push (`GBD-PCG/` is fingerprinted too since the 2026-08 fold).
The suite also runs GBD-PCG's own gate runner, so the receipt attests the cooperative solver too
(GBD-PCG carries no separate receipt — its standalone repo is retired).

⚠ `tools/regen_grid.py` regenerates against the **sibling GATO checkout's URDF** (same bytes as
`tools/iiwa14.urdf` but sitting next to the link STLs): the collision spherization resolves
meshes relative to the URDF file, and the lone vendored copy silently degrades the sphere set,
breaking the byte-diff gate. Since the CL-2b era grid.cuh bakes `grid_collision` + the EE
contact-frame map (unused by MPCGPU, carried for byte-identity) — build lines need
`-IGRiD/grid_codegen/collision` (already in the Makefile + tools scripts).

`tools/run_gates.sh` is the one-command gate runner (build + run + PASS/FAIL): single_cost_test
(one-solve EE-cost response at the fair config: ‖d_xu step‖ ≈ 339.6, post-solve window tracking
≈ 0.0107), test_terminal_cost (terminal q-gradient vs pinocchio truth on the committed
`tools/data/` dump inputs — the terminal-cost-bug regression test, see below), and one
`validate_track` tracking pass per linsys (PCG w/ GATO_REG_PATTERN ≈ 0.0288 mean, QDLDL ≈ 0.0294; EE-frame era).
GBD-PCG has its own `test/run_gates.sh`. Additional standing gates:

- `make test_fd_parity` — adapter `forwardDynamics` vs `grid::forward_dynamics_device` (expect max err 0).
- `grid.cuh` byte-diff vs GATO's iiwa14 grid.cuh (URDF-divergence gate).
- `GBD-PCG/examples/test_spd.exe` — SPD block-tridiagonal residual gate for the cooperative solver.
- Primitive migrations were validated **bit-exact** vs the hand-rolled originals.

## ⚠ Terminal-cost shared-memory aliasing bug (fixed 88c3853)

`kkt.cuh generate_kkt_submatrices` carved the last block's smem with `s_Qk = s_eePos_traj + 6`,
but the last block holds **two** 6-wide references (knots k and k+1) → the terminal reference
aliased the bottom of `s_Qk` and the terminal q-gradient came out wrong (dumped g row 63:
`-3.27` where pinocchio says `+6.31`). Fix: offset `6 → 2*6`. Regression test =
`tools/test_terminal_cost.cu` (4 kernels — kkt-layout replay, direct, shifted-arena, sequential —
vs a pinocchio-derived truth vector on the dumped solve-3000 inputs; run via `tools/run_gates.sh`).
All benchmark numbers predating the fix (before 2026-07-06) are invalid — do not mix.

## Benchmark (SSOT: `docs/benchmark_3way_2026-08-01.md`; the 07-06 doc keeps the conditioning analysis)

The 3-way iiwa14 fig8 benchmark config (2026-07-07): SQP=1, PCG cap 200, rel tol 1e-4,
RHO_INIT=0.01, **`-DGATO_REG_PATTERN`** — rho added only to the position half of Q, R
unregularized (GATO's convention; guarded in `include/{pcg,qdldl}/linsys_setup.cuh`). Under it
the stair preconditioner is near-ideal (cond(Pinv·S) ≈ 2e2), the native eta-exit is honest,
avg 1.1 PCG iters/solve → **0.222 ms/solve, tracking 0.0288** (EE-frame era 2026-08-01; fastest solver in the table at
B=1). The **default compile behavior** (no flag) is the historic full-Q+R regularization:
cond ≈ 3e4 and the eta-exit under-reports the true residual ~500x (fires ~10x early). Harnesses:
`tools/run_3way_iiwa.sh` (fair 3-way tracking check), `tools/time_persolve.sh [N] [pcg|qdldl]`
(isolated per-solve timing at any KNOT_POINTS), `tools/run_gates.sh` (correctness gates).

## ⚠ Corrected dynamics + the tracking benchmark

The regenerated grid is **pinocchio-exact**; the *old vendored* grid had a ~2×-wrong mass matrix. The
correct iiwa is **stiff** (last-joint inertia ≈ 0.003 → `Minv[6,6] ≈ 392`). The shipped `examples/trajfiles/`
were generated for the *wrong* robot, so the MPC is closed-loop **unstable** on them. Tracking on the OLD
shipped trajfiles is NOT a gate; tracking on the regenerated fig8 reference IS one now (post terminal-cost
fix: ≈ 0.0288 mean at the fair config, EE-frame era, checked by `tools/run_gates.sh`).

**Why it diverged, and the fix (diagnosed 2026-06-28).** The iiwa is 7-DOF tracking a 3-DOF EE-*position*
task → a 4-D cost nullspace that includes **joint 7** (its EE-position Jacobian column is ~0, and `s_Q[q-block]
= 0` so its position is uncosted). On the stiff robot joint 7 is a high-gain free integrator. A reference that
**commands joint motion** — as the old `gen_reference.cu` joint-space sinusoid did, with the *largest* sweep on
joint 7 — is untrackable by an EE-only cost → the nullspace runs away → **even the exact QDLDL solve diverges**
(the cooperative PCG just NaNs instead, harder divergence; it is NOT a PCG or co-residency bug — this GPU's
co-resident cap is 850 blocks ≫ N). Regulation/hold tracks ~0 fine; only commanded nullspace motion diverges.
GATO tracks fig8 with the *identical* cost because its reference is a **pure EE-space figure-8** that never
commands joints. So `tools/gen_reference.cu` now generates an **EE-space figure-8 reference + a constant
gravity-comp hold warm-start** (mirrors GATO's `run_mpc_fig8`); `make gen_ref`, then
`./tools/gen_reference.exe examples/trajfiles/0_0 <amp_scale> <period_s>` (amp_scale 0 ⇒ regulation). The
reference dt is locked to `TIMESTEP` (settings.cuh, default 0.01, shared with the tracker). Validate with
`tools/validate_track.cu` (single run, prefix arg, reports mean/max/final tracking).

Cost notes (`iiwa_eepos_plant.cuh`): the EE-position cost Hessian is the true Gauss-Newton `JᵀJ` (the legacy
code used the rank-1 gradient outer product `(Jᵀe)(Jᵀe)ᵀ`, fixed). `Q_COST` (default 0) adds joint-posture
regularization toward `q_nom=0` to anchor the EE-nullspace joints. See `docs/modernization.md`.

## Conventions

Short single-line commits. Hold pushes / upstream PRs until explicitly cleared. **NEVER run timing
concurrently with other box load** — timing needs an isolated quiet box, full stop. Correctness-only
runs are okay under load, but keep builds capped (sequential nvcc; no parallel compile fan-out).
