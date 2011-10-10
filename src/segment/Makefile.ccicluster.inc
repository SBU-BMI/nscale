CXX=g++
CXXFLAGS=-fPIC -pg -O3 -fopenmp #-g -msse4.2 
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CUDA_CV_INCLUDE =-I/opt/local/src/OpenCV-2.3.0/modules/gpu/src -I/opt/local/src/OpenCV-2.3.0/modules/gpu/src/cuda -I/usr/local/include -I/usr/local/include/opencv2/gpu
CXXFLAGS	+= -I/usr/local/include -I/usr/local/include/opencv -I../features
CXXFLAGS	+= ${CUDA_CV_INCLUDE} 
LDFLAGS		+= -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann 
#-L../features/execEngine -lexecengine
LDFLAGS		+= -lopencv_gpu


