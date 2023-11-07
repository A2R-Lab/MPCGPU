# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -IGLASS  -IGBD-PCG/include   -lcublas


examples: examples/main.exe 

examples/main.exe:
	$(NVCC) $(CFLAGS) examples/main.cu -o examples/main.exe

clean:
	rm -f examples/*.exe
