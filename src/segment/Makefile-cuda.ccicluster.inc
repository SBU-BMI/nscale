NVCC=nvcc
NVCC_FLAGS=--ptxas-options=-v -m64 -g -arch=sm_20 -O3 -Xcompiler -fPIC # -G # -keep -Xptxas -dlcm=cg --maxrregcount=20
NVCC_INCLUDE=-I/usr/local/cuda/include
#NVCC_INCLUDE=-I../NVIDIA\ GPU\ Computing\ SDK/C/common/inc/ -I/usr/local/cuda/include
CUDA_LIBS= -L/usr/local/cuda/lib64 -lcudart

CXXFLAGS	+= ${NVCC_INCLUDE}
CXXFLAGS	+= -DHAVE_CUDA
LDFLAGS		+= ${CUDA_LIBS}



