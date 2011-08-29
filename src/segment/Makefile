all: nu-segment.exe imrecontest.exe localmaxtest.exe misctest.exe imfilltest.exe

CXX=g++ 
CXXFLAGS=-g -pg -msse4.2 -fopenmp -O3
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

NVCC=nvcc
NVCC_FLAGS=-m64 -g -G
NVCC_INCLUDE=-I/home/tcpan/PhD/path/ext/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I/usr/lib/nvidia-cuda-toolkit/include/ -I/usr/include/ 
CUDA_LIBS= -L/usr/lib -lcudart

CUDA_CV_INCLUDE =-I/home/tcpan/PhD/path/ext/OpenCV-2.3.0/modules/gpu/src/

RM=/bin/rm -rf

CUDA_SRC = HistologicalEntitiesGPU.cpp MorphologicOperationsGPU.cpp

SRC= nu-segment.cpp HistologicalEntities.cpp MorphologicOperations.cpp ScanlineOperations.cpp 
IMRECON_SRC = imreconTest.cpp MorphologicOperations.cpp  ScanlineOperations.cpp
LOCALMAX_SRC = localMaximaTest.cpp MorphologicOperations.cpp  ScanlineOperations.cpp
MISC_SRC = miscTest.cpp MorphologicOperations.cpp  ScanlineOperations.cpp
IMFILL_SRC = imfillTest.cpp MorphologicOperations.cpp  ScanlineOperations.cpp


SRCO= ${SRC:.cpp=.o}
IMRECON_SRCO= ${IMRECON_SRC:.cpp=.o}
LOCALMAX_SRCO= ${LOCALMAX_SRC:.cpp=.o}
MISC_SRCO= ${MISC_SRC:.cpp=.o}
IMFILL_SRCO= ${IMFILL_SRC:.cpp=.o}

CUDA_SRCO = ${CUDA_SRC:.cpp=.o}

CXXFLAGS	+= `pkg-config opencv --cflags`
CXXFLAGS	+= ${NVCC_INCLUDE}
CXXFLAGS	+= ${CUDA_CV_INCLUDE}
LDFLAGS		+= `pkg-config opencv --libs`
LDFLAGS		+= -lopencv_gpu
LDFLAGS		+= ${CUDA_LIBS}


nu-segment.exe: ${SRCO} ${CUDA_SRCO} 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o nu-segment.exe ${SRCO} ${CUDA_SRCO}
imrecontest.exe: ${IMRECON_SRCO} ${CUDA_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o imrecontest.exe ${IMRECON_SRCO} ${CUDA_SRCO}
localmaxtest.exe: ${LOCALMAX_SRCO} ${CUDA_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o localmaxtest.exe ${LOCALMAX_SRCO} ${CUDA_SRCO}
misctest.exe: ${MISC_SRCO} ${CUDA_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o misctest.exe ${MISC_SRCO} ${CUDA_SRCO}
imfilltest.exe: ${IMFILL_SRCO} ${CUDA_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o imfilltest.exe ${IMFILL_SRCO} ${CUDA_SRCO}

clean:
	$(RM) *.o *.exe
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) -o $@ -c $<
	