# MPCGPU

Numerical experiments and the open-source solver from the paper ["MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU"](https://arxiv.org/abs/2309.08079) 

### Running examples

```
git clone https://github.com/A2R-Lab/MPCGPU.git
make
mkdir /tmp/results
./examples/pcg.exe
./examples/qdldl.exe
```

The track_iiwa_pcg file outlines how you could setup a tracking problem with underlying linear system solver as pcg.

The track_iiwa_qdldl file outlines how you could setup a tracking problem with underlying linear system solver as qdldl.

### Setting parameters

You can set a bunch of parameters in `include/setting.cuh` file. You can also modify these by passing them as
compiler flags. This will overwrite the default values set for these parameters. Please refer to `Makefile` for
an example.

### Other solvers and problems

You should be able to replace the underlying linear system solver with your own solver. Please refer to `include/linsys_solvers/qdldl/sqp.cuh` for an example.

You should also be able to compile and run it for a different problem that  "Kuka IIWA manipulator". Please refer to `include/rbdfiles/` folder for an example. We use [GRiD](!https://github.com/robot-acceleration/GRiD)  for computing rigid body dynamics with analytical gradients.

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
