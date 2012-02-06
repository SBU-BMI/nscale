CXX=/usr/bin/g++-4.4 
CXXFLAGS=-fPIC -pg -g -fopenmp -O3 #-g -msse4.2 -O3
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CXXFLAGS	+= -I/opt/local/include -I/opt/local/include/opencv -I../features-cpu -I../segment
LDFLAGS		+= -L/opt/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -L../features-cpu -lfeatures -L../segment -lsegment -L/usr/local/lib -lhdf5 -lhdf5_hl
#-lsz

