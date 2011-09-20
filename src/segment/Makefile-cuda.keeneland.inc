NVCC=nvcc
NVCC_FLAGS=--ptxas-options=-v -m64 -g -O3 #-G
NVCC_INCLUDE=-I/sw/keeneland/cuda/4.0/linux_binary/include
CUDA_LIBS= -L/sw/keeneland/cuda/4.0/linux_binary/lib64 -lcudart

CXXFLAGS	+= ${NVCC_INCLUDE}
CXXFLAGS	+= -DHAVE_CUDA
LDFLAGS		+= ${CUDA_LIBS}
