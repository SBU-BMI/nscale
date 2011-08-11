all: downhill.exe

CXX=gcc 
CXXFLAGS=-g -pg -O3 -msse4.2 -fopenmp
#CXX=icpc
#CXXFLAGS=-g -fast -openmp -msse4.2


RM=/bin/rm -rf

SRC= main.c VSGDownhill.c VSGFileImage.c 

SRCO= ${SRC:.c=.o}

CXXFLAGS = 
LDFLAGS = 

downhill.exe: ${SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o downhill.exe ${SRCO}

clean:
	$(RM) *.o *.exe
	
%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	
