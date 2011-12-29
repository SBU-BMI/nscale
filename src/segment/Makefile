include Makefile.inc
## uncomment below for keeneland.
#include Makefile.keeneland.inc
include Makefile-mpi.inc

## BEGIN CUDA SETTINGS - comment out to disable CUDA
include Makefile-cuda.inc
#include Makefile-cuda.keeneland.inc
CUDA_SRC = cuda/global_queue.cu cuda/neighbor-ops.cu cuda/watershed-ca-korbes.cu cuda/watershed-dw-korbes.cu cuda/queue.cu cuda/imrecon_queue_int_kernel.cu cuda/pixel-ops.cu cuda/change_kernel.cu cuda/imreconstruct_binary_kernel.cu cuda/imreconstruct_int_kernel.cu cuda/imreconstruct_float_kernel.cu
## END CUDA SETTINGS

#LIB_SRCH = HistologicalEntities.h MorphologicOperations.h PixelOperations.h utils.h
LIB_SRC = HistologicalEntities.cpp MorphologicOperations.cpp PixelOperations.cpp NeighborOperations.cpp FileUtils.cpp
LIB_SRCO = ${LIB_SRC:.cpp=.o}
EXECENG_SRC = SegmentTask.cpp
EXECENG_SRCO = ${EXECENG_SRC:.cpp=.o}

gputests: tests test/src/watershedTest.exe test/src/imreconChunkTest.exe test/src/imreconOptimizeProfiler.exe test/src/queueTest.exe
GPU_SRC = HistologicalEntitiesGPU.cpp MorphologicOperationsGPU.cpp PixelOperationsGPU.cpp NeighborOperationsGPU.cpp
GPU_LIB_SRC = ${CUDA_SRC} ${GPU_SRC}
GPU_LIB_SRCO = ${CUDA_SRC:.cu=.o} ${GPU_SRC:.cpp=.o}

all: libsegment.so ${EXECENG_SRCO} nu-segment.exe nu-segment-mpi-bot.exe nu-segmentExecEngine.exe nu-segmentChunk.exe

tests: all test/src/nu-segmentChunkTest.exe test/src/imreconTest.exe test/src/localMaximaTest.exe test/src/miscTest.exe test/src/imfillTest.exe test/src/bwareaopenTest.exe test/src/nu-segmentTest.exe test/src/nu-segmentProfile.exe



libsegment.so: ${LIB_SRCO} ${GPU_LIB_SRCO}
	$(CXX) ${CXX_FLAGS} $(LDFLAGS) -shared -o libsegment.so ${LIB_SRCO} ${GPU_LIB_SRCO}

clean:
	$(RM) libsegment.so *.o *.exe cuda/*.o *.cudafe?.* *.fatbin* *.cubin *.ptx *.i *.ii *.cu.cpp *.hash
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) ${CUDA_CV_INCLUDE} -o $@ -c $<
	
%.exe:  %.o ${EXECENG_SRCO}
	$(CXX) $(CXXFLAGS) -L. -lsegment -I. $(LDFLAGS) -o $@ ${EXECENG_SRCO} $< 

	
