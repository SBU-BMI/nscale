include Makefile.inc
## uncomment below for keeneland.
#include Makefile.keeneland.inc

## BEGIN MPI SETTINGS - comment out to disable MPI. overrides Makefile.inc for certain settings such as CXX
#include ../Makefile-mpi.inc
#include Makefile-mpi.keeneland.inc
## END MPI SETTINGS


## BEGIN CUDA SETTINGS - comment out to disable CUDA
include ../Makefile-cuda.inc
#include Makefile-cuda.keeneland.inc
## END CUDA SETTINGS

TASK_SRC = S1Task.cpp B1Task.cpp B2Task.cpp A4Task.cpp B3Task.cpp B4Task.cpp B5Task.cpp B6Task.cpp C1Task.cpp C2Task.cpp C3Task.cpp
TASK_O = ${TASK_SRC:.cpp=.o}

all: nu-segtasks-mpi-bot.exe

clean:
	$(RM) *.o *.exe *.so
	
%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c %.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

libhpdctasks.so: ${TASK_O}
	$(CXX) ${CXX_FLAGS} $(LDFLAGS) -shared -o libhpdctasks.so ${TASK_O}

%.exe: %.o libhpdctasks.so 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -L. -lhpdctasks -I. -o $@ $< 
