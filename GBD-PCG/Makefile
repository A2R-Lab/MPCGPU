NVCC = nvcc
CFLAGS = -Iinclude -IGLASS

SRCS = src/interface.cu src/pcg.cu src/utils.cu src/constants.cu
OBJS = $(SRCS:.cu=.o)
LIBRARY = libgpupcg.a

all: $(LIBRARY)

$(LIBRARY): $(OBJS)
	ar rcs $(LIBRARY) $(OBJS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIBRARY)