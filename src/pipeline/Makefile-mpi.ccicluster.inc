CXX=/usr/bin/mpic++ 

CXXFLAGS	+= -I/usr/lib64/openmpi/1.4-gcc/include
LDFLAGS		+= -L/usr/lib64/openmpi/1.4-gcc/lib -lmpich

CXXFLAGS	+= -DWITH_MPI
