# Makefile

# Compiler and compiler flags
NVCC = nvcc
CCBIN = /usr/bin/g++-10  # Adjust this to your GCC compiler path
CFLAGS = --compiler-options "-Wall -O3" -Iinclude -Iinclude/common -IGLASS -IGBD-PCG/include -Iqdldl/include -Lqdldl/build/out -lqdldl -lcublas

examples: examples/pcg.exe examples/qdldl.exe

examples/pcg.exe:
	$(NVCC) -ccbin $(CCBIN) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe

examples/qdldl.exe:
	$(NVCC) -ccbin $(CCBIN) $(CFLAGS) -DLINSYS_SOLVE=0 examples/track_iiwa_qdldl.cu -o examples/qdldl.exe

build_qdldl:
	cd qdldl && mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build . && cd ../../

clean:
	rm -f examples/*.exe
