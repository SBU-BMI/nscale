include Makefile.inc

## BEGIN CUDA SETTINGS - comment out to disable CUDA
include Makefile-cuda.inc
#include Makefile-cuda.keeneland.inc
CUDA_SRC = ccl.cu cudaScan.cu label512.cu label1024.cu label2048.cu
CUDA_SRCO = ${CUDA_SRC:.cu=.coo} 
## END CUDA SETTINGS

LIB_SRC = cclBase.cpp ccl.cpp cudaArray.cpp cudaBasicUtils.cpp cudaBuffer.cpp cudaScan.cpp cvlabeling_imagelab.cpp profiler.cpp
LIB_SRCO = ${LIB_SRC:.cpp=.o}


all: main.exe main2.exe


clean:
	$(RM) libgpuccl.so *.coo *.o *.exe *.cudafe?.* *.fatbin* *.cubin *.ptx *.i *.ii *.cu.cpp *.hash
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.coo: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) ${CUDA_CV_INCLUDE} -o $@ -c $<

#libgpuccl.so: ${LIB_SRCO} ${CUDA_SRCO}
#	$(CXX) ${CXX_FLAGS} $(LDFLAGS) -shared -o libgpuccl.so ${LIB_SRCO} ${CUDA_SRCO}

%.exe:  %.o ${LIB_SRCO} ${CUDA_SRCO} 
	$(CXX) $(CXXFLAGS) -I. $(LDFLAGS) -o $@ $< ${LIB_SRCO} ${CUDA_SRCO}
#	$(CXX) $(CXXFLAGS) -L. -lgpuccl -I. $(LDFLAGS) -o $@ $< 

	
