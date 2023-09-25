# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_86  -O3 -I. -IGPU-PCG/include -IGLASS  -Iqdldl/include -lcublas   -DTIME_LINSYS=1

# Name of the output executable
EXECUTABLE = examples/runme.exe

# Source file
SOURCE = examples/runme.cu

all:
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
