CXX=/usr/bin/g++-4.4 
CXXFLAGS=-fPIC -pg -g -fopenmp #-g -msse4.2 -O3
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

CXXFLAGS	+= `pkg-config opencv --cflags` -I../features-cpu -I../segment -I/usr/include
LDFLAGS		+= `pkg-config opencv --libs` -L../features-cpu -lfeatures -L../segment -lsegment -L/usr/lib -lhdf5 -lhdf5_hl

