# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_86  -O3 -Iinclude -IGPU-PCG/include -IGLASS  -lqdldl  -Iqdldl/include -Lqdldl/build/out -lcublas

examples: examples/pcg.exe examples/qdldl.exe

examples/pcg.exe:
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe
examples/qdldl.exe:
	$(NVCC) $(CFLAGS) -DPCG_SOLVE=0 examples/track_iiwa_qdldl.cu -o examples/qdldl.exe

clean:
	rm -f examples/*.exe
