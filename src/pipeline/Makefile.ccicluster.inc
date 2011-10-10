CXX=/usr/bin/g++-4.4 
CXXFLAGS=-fPIC -pg -g -fopenmp #-g -msse4.2 -O3
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CXXFLAGS	+= -I/usr/local/include -I/usr/local/include/opencv -I../features-cpu -I../segment
LDFLAGS		+= -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -L../features-cpu -lfeatures -L../segment -lsegment

