# MPCGPU

Numerical experiments and the open-source solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079) 

### Building and running examples

```
git clone https://github.com/A2R-Lab/MPCGPU
cd MPCGPU
make submodules          # git submodule update --init --recursive
make build_qdldl
make examples            # GPU arch defaults to sm_120; override with e.g. `make ARCH=sm_86 examples`
mkdir -p tmp/results
```
Either install the qdldl shared library by running ```cd qdldl/build && make install``` or modify the ```LD_LIBRARY_PATH``` environment variable to include the path to ```MPCGPU/qdldl/build/out```.

Run from the repo root (the examples read `examples/trajfiles/` and write `tmp/results/` via relative paths):

```
LD_LIBRARY_PATH=$PWD/qdldl/build/out ./examples/pcg.exe
LD_LIBRARY_PATH=$PWD/qdldl/build/out ./examples/qdldl.exe
```

Dependencies are vendored as submodules: [GRiD](https://github.com/robot-acceleration/GRiD) (rigid-body-dynamics code generation) and [GLASS](https://github.com/A2R-Lab/GLASS) (in-block GPU linear algebra); `make submodules` pulls everything. [GBD-PCG](GBD-PCG/) (the cooperative, grid-wide block-tridiagonal PCG solver) lives in-tree since 2026-08 — its standalone repo is retired, history preserved here — and shares the single top-level GLASS pin.

### Setting parameters

You can set a bunch of parameters in the `include/common/settings.cuh` file. They are all `#ifndef`-guarded, so you
can also override them by passing them as compiler flags (`-DNAME=value`) without editing the file. Please refer to
`Makefile` for an example.

### Reference trajectories

The examples track references from `examples/trajfiles/`. `tools/gen_reference.cu` (`make gen_ref`) generates a
self-consistent reference for the current dynamics model — an end-effector-space figure-8 plus a gravity-compensation
hold warm start:

```
make gen_ref
./tools/gen_reference.exe examples/trajfiles/0_0 <amp_scale> <period_s>   # amp_scale 0 => regulation/hold
```

The `0_0`–`0_2` trajfiles are regenerated figure-8 references. The remaining shipped `*_traj.csv` files predate the
corrected dynamics model (they were generated for a different robot model, and have no matching `*_eepos.traj`), so
they are not usable tracking references — regenerate with `gen_reference` instead.

### Validation and benchmarks

`tools/run_gates.sh` builds and runs the correctness gates (cost/step response, terminal-cost gradient
regression, and one tracking pass per linear-system solver). `tools/run_3way_iiwa.sh` runs the fair 3-way
iiwa14 figure-8 tracking comparison and `tools/time_persolve.sh` the isolated per-solve timing; the
benchmark configuration and results are documented in `docs/benchmark_3way_2026-08-01.md`. The in-tree
GBD-PCG solver has its own gate runner (`GBD-PCG/test/run_gates.sh`).

### Other solvers and problems

You should be able to replace the underlying linear system solver with your own solver. Please refer to `include/qdldl/sqp.cuh` for an example.

You should also be able to compile and run it for a different problem than the Kuka IIWA manipulator. Please refer to the `include/dynamics/` folder for an example. We use [GRiD](https://github.com/robot-acceleration/GRiD) for computing rigid body dynamics with analytical gradients; `include/dynamics/iiwa/grid.cuh` is generated from the pinned GRiD submodule via `make regen` (`tools/regen_grid.py`).

### Citing
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
