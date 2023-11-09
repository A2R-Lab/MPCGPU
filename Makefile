# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -IGLASS  -IGBD-PCG/include   -lcublas


examples: examples/double_pendulum.exe examples/main.exe  examples/double_integrator.exe examples/double_integrator_2d.exe

examples/main.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=9 -DNC=9 examples/main.cu -o examples/main.exe 

examples/double_pendulum.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=5 -DNUM_THREADS=64 -DNX=50 -DNC=50 examples/double_pendulum.cu -o examples/double_pendulum.exe 

examples/double_integrator.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=30 -DNC=30 examples/double_integrator.cu -o examples/double_integrator.exe 

examples/double_integrator_2d.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=30 -DNC=30 examples/double_integrator_2d.cu -o examples/double_integrator_2d.exe 

clean:
	rm -f examples/*.exe
