# GBD-PCG

Cooperative, GPU-wide preconditioned conjugate gradient for symmetric positive
definite block-tridiagonal systems. Maintained **in-tree in MPCGPU**, using its
top-level GLASS submodule for block primitives. No GATO dependency.

One CUDA block owns one knot/block row; cooperative grid synchronization joins
the iterations. This differs from GLASS's single-block PCG and is intentional.
All launched blocks must be simultaneously resident on the current GPU.

## Build and correctness examples

From the MPCGPU root, install the tooling and initialize submodules as in the
[root quickstart](../README.md), then:

```bash
make -C GBD-PCG/examples test ARCH=sm_120 STATE_SIZE=14 KNOT_POINTS=32
GBD-PCG/test/run_gates.sh
```

The examples use valid SPD known-solution systems, with float and double
variants. The pytest suite also covers small/odd dimensions, N=1, matvec,
nontrivial preconditioning, zero RHS, exact warm starts, iteration caps and
invalid configurations. It does not depend on ambient dumped files.

## API contract

Include `gpu_pcg.cuh` with `-IGBD-PCG/include -IGLASS` and C++17.
Matrix strips are column-major `[L | D | R]` per block row, totaling
`3 * state_size * state_size * knot_points` scalars. Boundary strips should
be zero. Vectors contain `state_size * knot_points` scalars.

- Host `solvePCG(A,b,x,s,N,&config)`: copies inputs, initializes identity Pinv,
  solves synchronously, copies only x back, returns actual iteration count.
  It supports `empty_pinv=true` only. A supplied host preconditioner/CSR input
  is unsupported and throws instead of running with uninitialized data.
- Device `solvePCGChecked(...)`: caller owns matrix, Pinv, RHS, solution and
  scratch allocations. Returns `{iterations, iteration_limit}`. The compatibility
  device `solvePCG` returns iterations. Neither is asynchronous.
- Runtime dimensions must equal compile-time STATE_SIZE/KNOT_POINTS.
  `pcg_block` must be one-dimensional, within device limits; launch occupancy
  is checked on the current device. Grid size is always N; legacy `pcg_grid`
  does not override the cooperative decomposition.
- Finite, nonnegative tolerances are required. Host inputs are finite-checked;
  device callers must supply valid finite SPD systems/preconditioners and
  non-overlapping buffers. Indefinite matrices are outside the contract:
  there is not yet a structured numerical-breakdown status.
- Default stopping uses `|eta| < abs_tol + rel_tol*|eta_initial|`, with
  `eta = rᵀ Pinv r`. This is **not** relative Euclidean residual tolerance.
  Exact zero residual exits safely even at zero tolerances.
- Experimental `PCG_TRUE_EXIT_CHECK_PERIOD` and
  `PCG_RESIDUAL_REPLACE_PERIOD` alter stopping/recurrence. They are not default
  performance claims; validate the resulting true residual for your system.

For higher-level ownership and limitations see [API notes](../docs/api.md).
The previous standalone repository is not needed to build this code.
