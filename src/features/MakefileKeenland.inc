CXX=g++
CXX_FLAGS = -fPIC -O3 # -fopenmp #-g -DVISUAL_DEBUG -O3 
OBJS=cooc.o cudaToCInterface.o Contour.o Blob.o RegionalMorphologyAnalysis.o DrawAuxiliar.o Operators.o

# optional way to get include dirs/libs path, etc `pkg-config --cflags opencv2 --libs opencv2`
#OPENCV_INCLUDE=/usr/local/include/opencv/
OPENCV_INCLUDE=-I/nics/c/home/gteodor/apps/OpenCV-2.3.0-install/include/ -I/nics/c/home/gteodor/apps/OpenCV-2.3.0-install/include/opencv
OPENCV_LIBS_DIR=-L/nics/c/home/gteodor/apps/OpenCV-2.3.0-install/lib

OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OPENCV_LIBS_GPU=${OPENCV_LIBS} -lopencv_gpu
EXAMPLES_DIR=./examples

NVCC=nvcc
NVCC_FLAGS=-Xcompiler -fPIC "-arch=sm_12"
NVCC_INCLUDE=-I../NVIDIA\ GPU\ Computing\ SDK/C/common/inc/ -I/usr/local/cuda/include
CUDA_LIBS= -L/nics/c/home/gteodor/apps/NVIDIA_GPU_Computing_SDK/C/lib -L/usr/local/cuda/lib64 -lcutil -lcuda # -lcudart


