
include ../Makefile.inc
CUDA_OBJS=cooc.o cudaToCInterface.o

all: $(CUDA_OBJS)
	cp -f *.o ..

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) -c $<

clean:
	rm -rf *.o
