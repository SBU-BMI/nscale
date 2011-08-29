
include ../Makefile.inc
CUDA_OBJS=cooc.o cudaToCInterface.o

all: $(CUDA_OBJS)
	cp -f *.o ..

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<
	echo %.o

clean:
	rm -rf *.o
