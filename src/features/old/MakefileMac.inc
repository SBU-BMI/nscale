CXX=g++
CXX_FLAGS = -fPIC -O3 # -fopenmp #-g -DVISUAL_DEBUG -O3 
CUDA_OBJS=cooc.o cudaToCInterface.o 
OBJS=Contour.o Blob.o RegionalMorphologyAnalysis.o DrawAuxiliar.o Operators.o $(CUDA_OBJS)

# optional way to get include dirs/libs path, etc `pkg-config --cflags opencv2 --libs opencv2`
OPENCV_INCLUDE=-I/usr/local/include/opencv/
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OPENCV_LIBS_GPU=${OPENCV_LIBS} -lopencv_gpu
EXAMPLES_DIR=./examples

NVCC=nvcc
#NVCC_FLAGS=-Xcompiler -fPIC "-arch=sm_11" 
NVCC_FLAGS=-m64 "-arch=sm_11" 
NVCC_INCLUDE=-I../NVIDIA\ GPU\ Computing\ SDK/C/common/inc/ -I/usr/local/cuda/include
CUDA_LIBS= -L/usr/local/cuda/lib -lcudart

EXEC_ENGINE_LIBS=${CUDA_LIBS}
