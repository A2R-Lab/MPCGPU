# GBD-PCG

> **Now part of MPCGPU.** GBD-PCG was folded into the
> [MPCGPU](https://github.com/A2R-Lab/MPCGPU) repo (this directory, full history
> preserved) in 2026-08; the standalone GBD-PCG repo is retired and no longer
> updated. In-block linear algebra comes from MPCGPU's top-level `GLASS/`
> submodule (single pin — no nested copy).

GBD-PCG is a **cooperative, grid-wide** preconditioned conjugate gradient solver for the
block-tridiagonal Schur-complement systems that arise in trajectory optimization. It is the
linear-system solver used by [MPCGPU](https://arxiv.org/abs/2309.08079). It solves

$P^{-1} S \lambda = P^{-1} \gamma$

where `S` and the preconditioner `Pinv` are symmetric block-tridiagonal — block dimension
`state_size`, matrix dimension `state_size * knot_points`.

## How it runs

The solver launches **one CUDA block per knot point** with `cudaLaunchCooperativeKernel` and
synchronizes the CG iteration across blocks with `grid.sync()` (cooperative-groups grid group),
so it requires a cooperative-launch-capable GPU and `knot_points` must fit in one co-resident
grid (the `checkPcgOccupancy` helper in `include/pcg.cuh` verifies this; the launch path does
not call it automatically). Each block owns one block-row of `S`/`Pinv` plus a halo
of its neighbours' state; cross-block dot-product reductions go through global scratch.

All in-block linear algebra defers to [GLASS](https://github.com/A2R-Lab/GLASS), MPCGPU's
top-level `GLASS/` submodule (`-I../GLASS` from this directory): the per-block-row band matvec `bdmv` is `glass::gemv`
(`include/utils.cuh`), dot products are `glass::dot_lowmem`, copies/reductions are
`glass::copy`/`glass::reduce`. There is no hand-rolled in-block BLAS here — new in-block
primitives belong upstream in GLASS.

## Matrix layout

`S` and `Pinv` are stored as per-block-row `[L|D|R]` strips:

- Block-row `i`'s strip starts at `i * 3 * state_size * state_size`.
- Within a strip the three `state_size x state_size` blocks are slot 0 = `L` (block `(i, i-1)`),
  slot 1 = `D` (diagonal), slot 2 = `R` (block `(i, i+1)`), each stored **column-major**
  (element `(r, c)` at `slot*d*d + c*d + r`).
- The absent boundary blocks (block 0's `L`, the last block's `R`) are **zero-filled**, and the
  matching halo vector slots are zeroed at load time, so every block runs one uniform full-width
  `[L|D|R]` matvec with no boundary special-casing.

`d_gamma` / `d_lambda` are dense length `state_size * knot_points` vectors.

## Requirements

- CUDA toolkit + a GPU with cooperative-launch support.
- MPCGPU's `GLASS` submodule: `make submodules` from the MPCGPU root.

Header-only to use: `#include "gpu_pcg.cuh"` with `-IGBD-PCG/include -IGLASS` (paths from the
MPCGPU root). `STATE_SIZE` and
`KNOT_POINTS` must be **compile-time constants** (`-DSTATE_SIZE=.. -DKNOT_POINTS=..`) — the
kernel is instantiated as `pcg<T, STATE_SIZE, KNOT_POINTS>`, and the runtime `state_size` /
`knot_points` arguments must match them.

## API

Configuration (`include/types.cuh`):

```c++
template <typename T>
struct pcg_config {
    T pcg_exit_tol;        // absolute floor on eta = r' * Pinv * r     (default 1e-6)
    T pcg_rel_tol;         // relative exit: |eta| < exit_tol + rel_tol*|eta_init| (default 1e-5)
    uint32_t pcg_max_iter; // iteration cap                              (default 25)
    dim3 pcg_grid;         // unused by the launch path (grid = knot_points)
    dim3 pcg_block;        // unused by the launch path (block = pcg_constants::DEFAULT_BLOCK = 64)
    int empty_pinv;        // legacy flag, only read by the host-array convenience overload
};
```

The main entry point (`include/interface.cuh`, via `gpu_pcg.cuh`) takes device pointers you
allocate — `d_S`/`d_Pinv` are the `3*state_size^2*knot_points` strip arrays, `d_r`/`d_p` are
`state_size*knot_points` scratch, `d_v_temp`/`d_eta_new_temp` are `knot_points` scratch — and
returns the iteration count:

```c++
template <typename T>
uint32_t solvePCG(const uint32_t state_size, const uint32_t knot_points,
                  T *d_S, T *d_Pinv, T *d_gamma, T *d_lambda,
                  T *d_r, T *d_p, T *d_v_temp, T *d_eta_new_temp,
                  pcg_config<T> *config);
```

Minimal usage (abridged from `examples/test_pcg_spd.cu`, the real correctness gate):

```c++
#include "gpu_pcg.cuh"   // build with -DSTATE_SIZE=6 -DKNOT_POINTS=8

pcg_config<float> config;            // defaults; tighten for a full solve:
config.pcg_exit_tol = 1e-8f;
config.pcg_rel_tol  = 1e-10f;
config.pcg_max_iter = STATE_SIZE * KNOT_POINTS * 4;

// fill d_S / d_Pinv ([L|D|R] strips, boundary strips zeroed), d_gamma,
// and d_lambda (initial guess) on device, then:
uint32_t iters = solvePCG<float>(STATE_SIZE, KNOT_POINTS,
                                 d_S, d_Pinv, d_gamma, d_lambda,
                                 d_r, d_p, d_v_temp, d_eta_new_temp, &config);
// solution is in d_lambda
```

Notes:

- A host-array convenience overload `solvePCG(h_S, h_gamma, h_lambda, state_size, knot_points,
  config)` exists but supports only `empty_pinv` and leaves `d_Pinv` uninitialized — use the
  device-pointer API above (as the tests and MPCGPU do).
- The `csr_t` overload is a stub (`NOT IMPLEMENTED`).

## Exit test and tolerance knobs

The exit test is **relative on the preconditioned residual** `eta = r'*Pinv*r`
(`|eta| < pcg_exit_tol + pcg_rel_tol*|eta_init|`), with a converged-start guard at iteration 0.
On ill-preconditioned systems eta can under-report the true residual `||gamma - S*lambda||` by
orders of magnitude (~500x at cond(Pinv·S) ≈ 3e4 on MPCGPU's historic regularization). Two
compile-time knobs in `include/pcg.cuh`, both `0` = off by default:

- `-DPCG_TRUE_EXIT_CHECK_PERIOD=K` — every K iterations test the TRUE residual
  `||gamma - S*lambda||^2 <= rel_tol^2 * ||gamma||^2` for the stop decision only.
- `-DPCG_RESIDUAL_REPLACE_PERIOD=K` — every K iterations recompute the true residual into the
  CG recurrence (classic residual replacement; fixes float32 recurrence drift only).

See `CLAUDE.md` for the full semantics and when each matters.

## Build and test

```bash
make submodules                # from the MPCGPU root (pulls GLASS)

# SPD residual gate at chosen dims (ARCH defaults to sm_120), from this directory:
make -C examples test STATE_SIZE=14 KNOT_POINTS=32

# One-command gate runner (run from this directory):
test/run_gates.sh              # or: ARCH=sm_86 test/run_gates.sh
```

`test/run_gates.sh` runs: `test_pcg_spd` (random SPD block-tridiagonal system, host residual
check) at 6x8 and 14x32, `test_bdmv` (strip matvec vs a host reference), and `test_pcg_dumped`
(standalone solve of a real dumped MPCGPU Schur system from `/tmp/mpc_{S,Pinv,gamma}.bin`;
skipped if the dumps are absent). `examples/pcg_solve.cu` is a legacy API-shape demo only — its
hardcoded system is degenerate and it does not validate anything.

## Citing

To cite this work in your research, please use the following bibtex:

```
@inproceedings{adabag2024mpcgpu,
  title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU},
  author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  address = {Yokohama, Japan},
  month={May.},
  year = {2024}
}
```
