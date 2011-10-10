CXX=g++ 
CXXFLAGS=-fPIC -pg -O3 -fopenmp # -g -msse4.2
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CUDA_CV_INCLUDE =-I/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/OpenCV-2.3.0/modules/gpu/src -I/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/OpenCV-2.3.0/modules/gpu/src/cuda -I/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/include/opencv2/gpu

CXXFLAGS	+= -I/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/include -I/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/include/opencv -I../features-cpu -I../segment
CXXFLAGS	+= ${CUDA_CV_INCLUDE}
LDFLAGS		+= -L/sw/keeneland/opencv/2.3.0/centos5.5_gnu4.4.0/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -L../features-cpu -lfeatures -L../segment -lsegment
LDFLAGS		+= -lopencv_gpu
