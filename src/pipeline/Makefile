include Makefile.inc
## uncomment below for keeneland.
#include Makefile.keeneland.inc

## BEGIN MPI SETTINGS - comment out to disable MPI. overrides Makefile.inc for certain settings such as CXX
include Makefile-mpi.inc
#include Makefile-mpi.keeneland.inc
## END MPI SETTINGS


## BEGIN CUDA SETTINGS - comment out to disable CUDA
#include Makefile-cuda.inc
#include Makefile-cuda.keeneland.inc
#CUDA_SRC = cuda/imrecon_queue_int_kernel.cu cuda/pixel-ops.cu cuda/change_kernel.cu cuda/imreconstruct_binary_kernel.cu cuda/imreconstruct_int_kernel.cu cuda/imreconstruct_float_kernel.cu
## END CUDA SETTINGS


all: nu-pipeline.exe nu-pipeline-omp.exe nu-pipeline-mpi.exe nu-pipeline-omp-mpi.exe

clean:
	$(RM) *.o *.exe
	
%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.exe:  %.o 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< 
%-omp.exe:  %.o 
	$(CXX) $(CXXFLAGS) -fopenmp $(LDFLAGS) -o $@ $< 
%-mpi.exe:  %.o 
	$(CXX) $(CXXFLAGS) -DWITH_MPI $(LDFLAGS) -o $@ $< 
%-omp-mpi.exe:  %.o 
	$(CXX) $(CXXFLAGS) -fopenmp -DWITH_MPI $(LDFLAGS) -o $@ $< 
		
