# Makefile

# Compiler and compiler flags
NVCC = nvcc

# GPU architecture (override on the command line, e.g. `make ARCH=sm_86`).
# Defaults to sm_120 (RTX 50-series / Blackwell). Replaces the old per-build -arch hand-edits.
ARCH ?= sm_120

CFLAGS = --compiler-options -Wall -O3 -DNDEBUG -arch=$(ARCH) -Iinclude -Iinclude/common -IGRiD/grid_codegen/collision -IGLASS -IGBD-PCG/include -lqdldl -Iqdldl/include -Lqdldl/build/out -lcublas


examples: examples/pcg.exe examples/qdldl.exe

examples/pcg.exe:
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe
examples/qdldl.exe:
	$(NVCC) $(CFLAGS) -DLINSYS_SOLVE=0 examples/track_iiwa_qdldl.cu -o examples/qdldl.exe

# Forward-dynamics adapter parity gate (adapter vs grid::forward_dynamics_device)
test_fd_parity:
	$(NVCC) $(CFLAGS) examples/test_fd_parity.cu -o examples/test_fd_parity.exe

# Self-consistent reference generator (grid.cuh FK+ID at the corrected robot). Run from repo root:
#   ./tools/gen_reference.exe examples/trajfiles/0_0 <amp_scale> <period_s>
# amp_scale 0 => regulation/hold; reference dt is locked to TIMESTEP in settings.cuh.
gen_ref:
	$(NVCC) $(CFLAGS) tools/gen_reference.cu -o tools/gen_reference.exe

# Pull GRiD/GLASS/GBD-PCG (and their nested submodules) to the pinned commits
submodules:
	git submodule update --init --recursive

# Regenerate include/dynamics/iiwa/grid.cuh from the pinned GRiD submodule (CPU, needs the GATO .venv
# or any env with the GRiDCodeGenerator deps). Output is byte-identical to GATO's iiwa14 grid.cuh.
regen:
	python tools/regen_grid.py

build_qdldl:
	cd qdldl && mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build . && cd ../../

clean:
	rm -f examples/*.exe
