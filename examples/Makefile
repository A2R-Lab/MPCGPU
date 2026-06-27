NVCC = nvcc
ARCH ?= sm_120
CFLAGS = --compiler-options -Wall  -O3 -I../include -I../GLASS -arch=$(ARCH)

# Dims for the SPD residual test (compile-time kernel template params).
STATE_SIZE ?= 14
KNOT_POINTS ?= 32

examples: pcg.exe pcg_dp.exe test_spd.exe


pcg.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe

pcg_dp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg_dp.exe

# SPD block-tridiagonal residual gate — the real correctness test (see test_pcg_spd.cu).
# Override dims: make test_spd.exe STATE_SIZE=6 KNOT_POINTS=8
test_spd.exe:
	$(NVCC) $(CFLAGS) -DSTATE_SIZE=$(STATE_SIZE) -DKNOT_POINTS=$(KNOT_POINTS) test_pcg_spd.cu -o test_spd.exe

test: test_spd.exe
	./test_spd.exe

clean:
	rm -rf *.exe
