# MPCGPU

Numerical experiments and the open-source solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079) 

### Building and running examples

```
git clone https://github.com/A2R-Lab/MPCGPU
cd MPCGPU
git submodule update --init --recursive
make build_qdldl
make examples
mkdir -p tmp/results
```
Either install the qdldl shared library by running ```cd qdldl/build && make install``` or modify the ```LD_LIBRARY_PATH``` environment variable to include the path to ```MPCGPU/qdldl/build/out```.

```
./examples/pcg.exe
./examples/qdldl.exe
```

### Setting parameters

You can set a bunch of parameters in the `include/common/settings.cuh` file. You can also modify these by passing them as
compiler flags. This will overwrite the default values set for these parameters. Please refer to `Makefile` for
an example.

### Validation and benchmarks

`tools/run_gates.sh` builds and runs the correctness gates (cost/step response, terminal-cost gradient
regression, and one tracking pass per linear-system solver). `tools/run_3way_iiwa.sh` runs the fair 3-way
iiwa14 figure-8 tracking comparison and `tools/time_persolve.sh` the isolated per-solve timing; the
benchmark configuration and results are documented in `docs/benchmark_3way_2026-07-06.md`.

### Other solvers and problems

You should be able to replace the underlying linear system solver with your own solver. Please refer to `include/qdldl/sqp.cuh` for an example.

You should also be able to compile and run it for a different problem that  "Kuka IIWA manipulator". Please refer to `include/dynamics/` folder for an example. We use [GRiD](!https://github.com/robot-acceleration/GRiD)  for computing rigid body dynamics with analytical gradients.

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
