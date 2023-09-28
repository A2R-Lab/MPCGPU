# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -IGLASS  -IGPU-PCG/include  -lqdldl  -Iqdldl/include -Lqdldl/build/out -lcublas


examples: examples/pcg.exe examples/qdldl.exe

examples/pcg.exe:
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe
examples/qdldl.exe:
	$(NVCC) $(CFLAGS) -DLINSYS_SOLVE=0 examples/track_iiwa_qdldl.cu -o examples/qdldl.exe

build_qdldl:
	cd qdldl && mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build . && cd ../../

clean:
	rm -f examples/*.exe
