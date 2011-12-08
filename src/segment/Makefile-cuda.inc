NVCC=nvcc
NVCC_FLAGS=--ptxas-options=-v -m64 -O3 -gencode arch=compute_20,code=sm_20 -Xcompiler -fPIC # -keep -Xptxas -dlcm=cg --maxrregcount=20 -g -G 
NVCC_INCLUDE=-I/usr/local/cuda/include
#NVCC_INCLUDE=-I../NVIDIA\ GPU\ Computing\ SDK/C/common/inc/ -I/usr/local/cuda/include
CUDA_LIBS= -L/usr/local/cuda/lib64 -lcudart

CXXFLAGS	+= ${NVCC_INCLUDE}
CXXFLAGS	+= -DHAVE_CUDA
LDFLAGS		+= ${CUDA_LIBS}



