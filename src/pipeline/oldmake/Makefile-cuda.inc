NVCC=nvcc
NVCC_FLAGS=-m64 --use_fast_math -arch=sm_20 -Xcompiler -fPIC -O3 -DPREFIX_SUM # -G -g #  --ptxas-options=-v -keep -Xptxas -dlcm=cg --maxrregcount=20
NVCC_INCLUDE=-I/usr/local/cuda/include
CUDA_LIBS= -L/usr/local/cuda/lib64 -lcudart

CXXFLAGS	+= ${NVCC_INCLUDE}
CXXFLAGS	+= -DHAVE_CUDA
LDFLAGS		+= ${CUDA_LIBS}



