# GBD-PCG + ADMM


### Building and running examples

We have the following examples:
- random QP
- double pendulum 
- double integrator (1D)
- double integrator (2D)

```
git clone https://github.com/A2R-Lab/MPCGPU
git checkout admm
git submodule update --init --recursive
make
./examples/double_pendulum.exe
```

### Setting parameters

You need to pass certain parameters as compile time constants, see the makefile of examples to understand this.

The parameters and their default values
- NX : 9
- NC : 9
- NUM_THREADS : 64
- KNOT_POINTS : 3
- STATE_SIZE (NX/KNOT_POINTS) : 3
- DEBUG_MODE : 0

Some constraints:
- KNOT_POINTS >= 3
- NUM_THREADS >= max(NX, NC)


### Citing
To cite this work in your research, please use the following bibtex:
```
@misc{adabag2023mpcgpu,
      title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU}, 
      author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
      year={2023},
      eprint={2309.08079},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
