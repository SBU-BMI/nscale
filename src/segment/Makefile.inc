CXX=/usr/bin/g++ 
CXXFLAGS=-fPIC  -fopenmp -O3 #-g -msse4.2 -fopenmp 
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf
OPENCV_SRC_INCLUDE_DIR=/usr/local/src/OpenCV-2.3.0/modules/gpu/src/

CUDA_CV_INCLUDE =-I${HOME}/apps/OpenCV-2.3.0/modules/gpu/src -I${HOME}/apps/OpenCV-2.3.0/modules/gpu/src/cuda -I/usr/local/include -I/usr/local/include/opencv2/gpu -I${OPENCV_SRC_INCLUDE_DIR}
CXXFLAGS	+= `pkg-config opencv --cflags` -I../ -DPRINT_CONTOUR_TEXT
CXXFLAGS	+= ${CUDA_CV_INCLUDE} 
LDFLAGS		+= `pkg-config opencv --libs` -L../execEngine  -lexecengine
LDFLAGS		+= -lopencv_gpu


