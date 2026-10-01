# Supported C++ contracts

## MPCGPU

The supported example is fixed-base iiwa14, 14 states/7 controls, float, fixed
horizon at compile time, EE-position cost, and fixed simulation pacing.
PCG runs on the GPU; QDLDL factorization runs on the CPU. CUDA cooperative
launch support and enough simultaneous block residency are necessary for PCG.
Other robots/contact/locomotion/batching are not supported by this release
candidate. Python is a codegen/test dependency, not a solver wrapper.

`mpcgpu::Reference<float>` reads paired `<prefix>_eepos.traj` (6 columns) and
`<prefix>_traj.csv` (21 columns), validates finite rectangular numeric rows,
equal row counts and at least the horizon length. The public tracking example
does this before GPU allocation. Tracking mean/max/final use Euclidean (L2)
EE-position error; old L1 final-error values are not interchangeable.

The header-based SQP entry points retain their existing arguments with an
optional final `mpcgpu::SqpWorkspace*`. Omit it for an owned temporary workspace.
The tracking driver reuses one workspace for all solves. It owns device and
host scratch, eight streams and a cuBLAS handle, and caches QDLDL's symbolic
structure. Numerical factorization and solve state are recomputed per call.

A workspace is bound to one backend, dimension tuple and current CUDA device.
It is non-copyable and **not thread-safe**: do not share across concurrent solves.
Inputs, solution/warm start and robot model are still caller/driver-owned.
Calls are synchronous; destruction releases owned resources. A dimensions,
device or allocation-layout mismatch throws. This is not yet a stable
cross-language ABI; some internal CUDA error helpers still terminate on errors.

## GBD-PCG

GBD-PCG is a pinned submodule; see [its README](../GBD-PCG/README.md) for matrix layout and entry points.
Host convenience has an identity preconditioner; the device API accepts Pinv
and caller-owned scratch. Float and double are tested. Invalid dimensions,
tolerances, unsupported CSR/host Pinv modes and cooperative capacity are rejected.
Device buffers must be valid finite SPD inputs; numerical breakdown for
indefinite systems is not a supported recovery API.

`iteration_limit=false` reports the configured stopping condition, **not**
certified true-residual accuracy. Default eta is a preconditioned recurrence
norm. Check the actual residual when accepting an application solution,
especially with weak preconditioners or poorly conditioned matrices.

## Configuration boundaries

The content-aware builder owns the reviewed correctness/timing configurations.
Correctness disables native solver timers and removes wall-clock stopping;
it must not be used to report speed. The timing profile preserves native SQP
timestamps. Those times exclude some public-call setup/cleanup: report them as
internal SQP time, not complete control-loop latency.

Build modes share mathematical flags; `MPCGPU_NO_REUSE` is a diagnostic fresh
workspace baseline. The shared builder selects position-only regularization
for the reviewed example configuration.

Shared orchestration across PCG/QDLDL still has duplication. Deliberately avoid
a generic backend framework during this safety update; further extraction
should preserve operation order and carry parity tests.
