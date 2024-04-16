# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -Iinclude/common -IGLASS  -IGBD-PCG/include  -lqdldl  -Iqdldl/include -Lqdldl/build/out -lcublas


examples: examples/pcg.exe examples/qdldl.exe examples/pcg_n.exe

examples/pcg.exe: examples/track_iiwa_pcg.cu
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe

examples/pcg_n.exe: examples/track_iiwa_pcg_n.cu
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg_n.cu -o examples/pcg_n.exe

examples/qdldl.exe: examples/track_iiwa_qdldl.cu
	$(NVCC) $(CFLAGS) -DLINSYS_SOLVE=0 examples/track_iiwa_qdldl.cu -o examples/qdldl.exe

build_qdldl:
	cd qdldl && mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build . && cd ../../

clean:
	rm -f examples/*.exe
