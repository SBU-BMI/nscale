CXX=/usr/bin/g++-4.4 
CXXFLAGS=-fPIC -pg -O3 -fopenmp #-g -msse4.2 -fopenmp 
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CUDA_CV_INCLUDE =-I/home/tcpan/PhD/path/ext/OpenCV-2.3.0/modules/gpu/src -I/home/tcpan/PhD/path/ext/OpenCV-2.3.0/modules/gpu/src/cuda -I/usr/local/include -I/usr/local/include/opencv2/gpu
CXXFLAGS	+= `pkg-config opencv --cflags` -I../features -DPRINT_CONTOUR_TEXT
CXXFLAGS	+= ${CUDA_CV_INCLUDE} 
LDFLAGS		+= `pkg-config opencv --libs` -L../features/execEngine -lexecengine
LDFLAGS		+= -lopencv_gpu


