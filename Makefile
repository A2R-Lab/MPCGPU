# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -IGLASS  -IGBD-PCG/include   -lcublas


examples: examples/double_pendulum.exe examples/qp.exe  examples/double_integrator_1d.exe examples/double_integrator_2d.exe examples/double_precision.exe

examples/qp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=9 -DNC=9 examples/qp.cu -o examples/qp.exe 

examples/double_pendulum.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=5 -DNUM_THREADS=64 -DNX=50 -DNC=50 examples/double_pendulum.cu -o examples/double_pendulum.exe 

examples/double_precision.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=5 -DNUM_THREADS=64 -DNX=50 -DNC=50 examples/double_precision.cu -o examples/double_precision.exe 


examples/double_integrator_1d.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=3 -DNUM_THREADS=64 -DNX=30 -DNC=30 examples/double_integrator_1d.cu -o examples/double_integrator_1d.exe 

examples/double_integrator_2d.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=6 -DNUM_THREADS=64 -DNX=60 -DNC=60 examples/double_integrator_2d.cu -o examples/double_integrator_2d.exe 

run:
	./examples/qp.exe
	./examples/double_integrator_1d.exe
	./examples/double_integrator_2d.exe
	./examples/double_pendulum.exe

clean:
	rm -f examples/*.exe
