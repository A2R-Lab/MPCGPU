# CLAUDE.md — orientation for AI agents (and humans) working on GBD-PCG

GBD-PCG is a **cooperative, grid-wide** preconditioned conjugate gradient solver
for the block-tridiagonal Schur systems that arise in optimal control. It solves
`Pinv · S · λ = Pinv · γ` where `S` (and the preconditioner `Pinv`) are symmetric
block-tridiagonal — block dimension `state_size`, matrix dimension
`state_size · knot_points`.

## The mental model — cooperative vs block-wide

GBD-PCG launches **one CUDA block per knot point** with
`cudaLaunchCooperativeKernel`, and the iteration is synchronized across blocks
with `grid.sync()` (cooperative-groups grid group). Each block owns one block-row
of `S`/`Pinv` plus a halo of its neighbours' state, and the cross-block dot-product
reductions go through global scratch (`d_v_temp` / `d_eta_new_temp`). This is what
lets the solver scale past a single block's resident capacity (many knots).

This is the **cooperative grid-wide analog** of GLASS's single-block
`glass::pcg::solve` (and `glass::bdmv`), which run the *entire* block-tridiagonal
system inside one block. Same math, different decomposition — pick GBD-PCG when the
problem is too large for one block; pick `glass::pcg` when it fits. **Do not** try
to replace GBD-PCG's cooperative loop with `glass::pcg` (it abandons the grid-wide
scaling), and **do not** push GBD-PCG's cross-block work into GLASS (GLASS is
single-block by charter: "never split a primitive across blocks").

## Defers in-block linear algebra to GLASS

The per-block (in-block) linear algebra is **GLASS** (`glass::`, submodule pinned
to match the consuming repo — currently `5caa6d0`, naming r2 + tile4 gemm):

- The per-block-row band matvec is `glass::gemv` (column-major, `ROW_MAJOR=false`)
  — see `bdmv` in `include/utils.cuh`. With the absent `L` (block 0) / `R` (last
  block) matrix strips zeroed in `pcg.cuh`'s populate phase and the absent halo pad
  slots zeroed in `loadbdVec`, every block runs ONE uniform full-width `[L|D|R]`
  matvec — no first/middle/last special-casing.
- Dot products use `glass::dot_lowmem(n, x, y, out)` (result in `out[0]`, inputs
  preserved — GBD-PCG reuses `s_r_tilde` after the dot, so the in-place destructive
  `glass::dot` is NOT usable here).
- Cross-block sum reductions = `glass::copy` the per-block partials into shared then
  in-place `glass::reduce(n, x)` (the old 3-arg `reduce(dest,n,src)` is gone).
- Vector copies = `glass::copy(n, src, dst)`.

So GBD-PCG carries no hand-rolled in-block BLAS. If you need a new in-block
primitive, add it upstream in GLASS, not here.

## Exit-test semantics (know this before trusting `iters`)

The solver's relative exit tests **eta = r'·Pinv·r from the CG recurrence**, not the true
residual `‖γ − Sλ‖`. On ill-preconditioned systems eta under-reports the true residual by
orders of magnitude (float64 shows the same → weak-norm property, not float drift): on
MPCGPU's iiwa14 Schur system under the historic full-Q+R regularization
(cond(Pinv·S) ≈ 3e4) it lies ~500x and fires ~10x early. Under MPCGPU's
`-DGATO_REG_PATTERN` benchmark config (cond ≈ 2e2) the same exit is honest (avg 1.1
iters/solve). See MPCGPU `docs/benchmark_3way_2026-07-06.md`. Two guarded experiment
knobs in `include/pcg.cuh`, both **0 = off by default** (pure upstream behavior):

- `PCG_TRUE_EXIT_CHECK_PERIOD=K` — every K iterations test the TRUE residual
  `‖γ − Sλ‖² ≤ rel_tol²·‖γ‖²` for the STOP decision only (recurrence untouched); the
  eta-based relative exit is disabled while on (iter-0 converged-start guard kept).
  Overhead ~half an iteration per K.
- `PCG_RESIDUAL_REPLACE_PERIOD=K` — every K iterations recompute the true residual INTO
  the recurrence (classic residual replacement; fixes float32 recurrence drift only, not
  the weak-norm under-reporting).

## Source layout

- `include/pcg.cuh` — the cooperative `pcg<T, state_size, knot_points>` kernel
  (grid.sync iteration). `include/interface.cuh` — host `solvePCG` wrappers +
  cooperative launch. `include/utils.cuh` — `bdmv` (GLASS gemv) + `loadbdVec` (halo
  gather with zero-padded boundaries). `include/{types,constants,gpuassert}.cuh`.
- `GLASS/` — submodule (the in-block linear-algebra library).
- `examples/` — `pcg_solve.cu` (legacy demo; **note**: it leaves `d_Pinv`
  uninitialized and its hardcoded `S` is indefinite, so it NaNs — kept for API
  shape only). **`test_pcg_spd.cu` is the real correctness gate**: random SPD
  block-tridiagonal `S` + identity `Pinv`, checks `‖γ − Sλ‖` on host. Run it with
  `make -C examples test` (override `STATE_SIZE`/`KNOT_POINTS`).

## Build & test

Header-only to *use*. To run the test (needs a cooperative-launch-capable GPU):

```bash
cd examples && make test STATE_SIZE=14 KNOT_POINTS=32   # ARCH defaults to sm_120
test/run_gates.sh          # one-command gate runner (from repo root): test_pcg_spd at
                           # 6x8 + 14x32, test_bdmv, test_pcg_dumped (skips if no dumps)
```

Validate migration changes with `test_pcg_spd.cu` (it exercises `bdmv` on block 0 /
middle / last every iteration) + `compute-sanitizer --tool memcheck|racecheck`.
`examples/test_bdmv.cu` isolates the strip matvec vs a host reference;
`examples/test_pcg_dumped.cu` runs the solver standalone on a REAL dumped Schur system
(`/tmp/mpc_{S,Pinv,gamma}.bin`, produced by an MPCGPU `validate_track` built with
`-DDUMP_KKT`).

## Conventions

- Short, single-line commit messages; no `Co-Authored-By` footer.
- Preserve the cooperative grid-wide structure; keep in-block linalg on `glass::`.
- Bump the `GLASS` submodule pin to match the consuming repo (MPCGPU / GATO).
