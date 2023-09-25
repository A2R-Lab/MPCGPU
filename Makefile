# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_86  -O3 -I. -IGPU-PCG/include -IGLASS  -lqdldl  -Iqdldl/include -Lqdldl/build/out -lcublas

# Name of the output executable
EXECUTABLE = examples/runme.exe

# Source file
SOURCE = examples/runme.cu

examples: examples/pcg.exe examples/qdldl.exe

examples/pcg.exe:
	$(NVCC) $(CFLAGS) $(SOURCE) -o examples/pcg.exe
examples/qdldl.exe:
	$(NVCC) $(CFLAGS) -DPCG_SOLVE=0 $(SOURCE) -o examples/qdldl.exe

clean:
	rm -f examples/*.exe
