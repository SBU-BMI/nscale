CXX=mpic++ 

CXXFLAGS	+= -I/usr/lib64/openmpi/1.4-gcc/include
LDFLAGS		+= -L/usr/lib64/openmpi/1.4-gcc/lib -lmpi

CXXFLAGS	+= -DWITH_MPI
