CXX=/usr/bin/mpic++ 

CXXFLAGS	+= -I/usr/include/mpich2
LDFLAGS		+= -L/usr/lib64 -L/usr/lib -lmpich -lcr

#CXXFLAGS	+= -DWITH_MPI
