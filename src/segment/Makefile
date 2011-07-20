all: test.exe imrecontest.exe

CXX=g++ 
#CXX=g++44
CXXFLAGS=-g -O3 -msse4.2 -fopenmp
#CXXFLAGS=-O2 -msse4.2 -fopenmp
#CXX=icc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

SRC= test.cpp HistologicalEntities.cpp MorphologicOperations.cpp 
MOP_SRC = imreconTest.cpp MorphologicOperations.cpp ImageOperations.cpp


SRCO= ${SRC:.cpp=.o}

MOP_SRCO= ${MOP_SRC:.cpp=.o}

CXXFLAGS	+= `pkg-config opencv --cflags` 
LDFLAGS		+= `pkg-config opencv --libs`


test.exe: ${SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o test.exe ${SRCO}
imrecontest.exe: ${MOP_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o imrecontest.exe ${MOP_SRCO}
	
clean:
	$(RM) *.o *.exe
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	