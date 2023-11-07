# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -IGLASS  -IGBD-PCG/include   -lcublas


examples: examples/main.exe 

examples/main.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=9 -DNC=9 examples/main.cu -o examples/main.exe 

clean:
	rm -f examples/*.exe
