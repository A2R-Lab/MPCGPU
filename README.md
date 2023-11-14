# GBD-PCG + ADMM

### Main interface function

```
admm_solve_outer(T * h_H,  T *h_g, T *h_A, T * h_l , T * h_u,  T * h_x,  T *h_lambda, T *h_z, float rho, float sigma =1e-6, float tol =1e-3, int max_iters=1000, int update_rho=1)
```

Input sizes:
- Matrices are expected to be in column major order
	- h_H (NX * NX)  
	- h_A (NC *NX)
- Vectors
	- h_g (NX)
	- h_l (NC)
	- h_u (NC)
	- h_x (NX)
	- h_lambda (NC)
	- h_z (NC)



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
