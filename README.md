# MPCGPU

CUDA/C++ nonlinear model predictive control with cooperative GPU-wide PCG,
from [MPCGPU (ICRA 2024)](https://arxiv.org/abs/2309.08079).
The maintained GBD-PCG implementation lives [in this repository](GBD-PCG/).
Dynamics come from **GRiD**; block linear algebra comes from **GLASS**.

The development branch modernizes the iiwa14 implementation, validation and
builds. The ICRA 2024 paper reports the original experiments; updated performance
measurements will follow validation in a dedicated timing window.

## Quickstart

Requires Linux, Python 3.11+, CMake, a C++17-capable CUDA toolkit, cuBLAS, and an
NVIDIA GPU supporting cooperative launches. Choose the CUDA architecture for
your GPU (the lab default below is `sm_120`). Builds are sequential.

```bash
git clone --recurse-submodules https://github.com/A2R-Lab/MPCGPU.git
cd MPCGPU
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
make build_qdldl
make ARCH=sm_120 examples
mkdir -p tmp/results
LD_LIBRARY_PATH="$PWD/qdldl/build/out:${LD_LIBRARY_PATH:-}" ./examples/pcg.exe
LD_LIBRARY_PATH="$PWD/qdldl/build/out:${LD_LIBRARY_PATH:-}" ./examples/qdldl.exe
```

Run from the repository root. These are **correctness-profile**, fixed-pacing
figure-eight tracking demos, not the paper's timing sweep. Both report finite
L2 end-effector position error (mean, maximum, final). Optional arguments are
a reference prefix and an output prefix. Invalid/missing trajectories fail
before GPU allocation. The build cache checks source, headers, dependency
pins, compiler, flags, architecture and binary contents.

### ICRA 2024 pick-and-place task

```bash
make icra BACKEND=pcg KNOT_POINTS=64      # or BACKEND=qdldl; horizons 32 to 512
```

This builds and runs the paper's five-goal circuit under its recovered protocol,
including zero gravity, then checks goal visits, final hold and tracking error.
Output lands in `tmp/icra/<backend>-N<knots>/` with a trajectory plot. It is a
correctness run with simulated 500 Hz control, not a timing measurement. See
[the replication notes](docs/icra-replication.md) for sources and differences.

## Validation and configuration

```bash
.venv/bin/python -m pytest test/test_host.py -q  # no GPU required
tools/run_gates.sh                             # GPU correctness, no timing
make check-codegen                            # pinned local model/recipe
make -C GBD-PCG/examples test                  # float/double SPD examples
```

The default build profile uses float MPC, horizon 64, one SQP step, PCG cap 200,
relative eta tolerance 1e-4, and position-only regularization with rho 0.01.
Header defaults differ: use the shared builder for reproducible examples.
Set `ARCH`, `KNOT_POINTS`, or `EXTRA_FLAGS` on make; changed settings rebuild.
GBD-PCG supports float and double; the QDLDL-backed MPC build uses float.

Generate a new reference without overwriting the shipped fixture:

```bash
make gen_ref
mkdir -p tmp/references
LD_LIBRARY_PATH="$PWD/qdldl/build/out" ./tools/gen_reference.exe tmp/references/fig8 0.15 6
LD_LIBRARY_PATH="$PWD/qdldl/build/out" ./examples/pcg.exe tmp/references/fig8
```

The supported shipped inputs are the paired `0_0`–`0_2` figure-eight references
and the ICRA circuit in `examples/icra/`. Other legacy trajectory files are
historical, not current model validation.

## Scope and documentation

MPCGPU provides CUDA/C++ solvers and fixed-base iiwa14 tracking examples.
Python supports code generation and validation. See the API notes for the
supported solver configurations and interfaces. GBD-PCG requires SPD systems
and cooperative co-residency; its relative eta criterion is not a true-residual
guarantee.

- [Current documentation and limitations](docs/README.md)
- [C++ ownership and solver contracts](docs/api.md)
- [Development and signed correctness receipts](docs/development.md)
- [Quiet-window timing handoff](docs/timing.md)
- [ICRA example replication](docs/icra-replication.md)
- [Audit and implementation plan](docs/audit-and-plan-2026-09-28.md)
- [Local project website](website/README.md)

## Citation and license

```bibtex
@inproceedings{adabag2024mpcgpu,
  title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU},
  author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

MPCGPU code, documentation, website and paper figures are [MIT licensed](LICENSE).
Dependency license notices are listed in [NOTICE](NOTICE).
