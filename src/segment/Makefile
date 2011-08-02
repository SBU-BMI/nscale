all: nu-segment.exe imrecontest.exe localmaxtest.exe misctest.exe imfilltest.exe

CXX=g++ 
#CXX=g++44
CXXFLAGS=-g -pg -O3 -msse4.2 -fopenmp
#CXXFLAGS=-O2 -msse4.2 -fopenmp
#CXX=icc
#CXXFLAGS=-g -fast -openmp -msse4.2

RM=/bin/rm -rf

SRC= nu-segment.cpp HistologicalEntities.cpp MorphologicOperations.cpp
IMRECON_SRC = imreconTest.cpp MorphologicOperations.cpp
LOCALMAX_SRC = localMaximaTest.cpp MorphologicOperations.cpp
MISC_SRC = miscTest.cpp MorphologicOperations.cpp
IMFILL_SRC = imfillTest.cpp MorphologicOperations.cpp


SRCO= ${SRC:.cpp=.o}

IMRECON_SRCO= ${IMRECON_SRC:.cpp=.o}
LOCALMAX_SRCO= ${LOCALMAX_SRC:.cpp=.o}
MISC_SRCO= ${MISC_SRC:.cpp=.o}
IMFILL_SRCO= ${IMFILL_SRC:.cpp=.o}

CXXFLAGS	+= `pkg-config opencv --cflags` 
LDFLAGS		+= `pkg-config opencv --libs`


nu-segment.exe: ${SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o nu-segment.exe ${SRCO}
imrecontest.exe: ${IMRECON_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o imrecontest.exe ${IMRECON_SRCO}
localmaxtest.exe: ${LOCALMAX_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o localmaxtest.exe ${LOCALMAX_SRCO}
misctest.exe: ${MISC_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o misctest.exe ${MISC_SRCO}
imfilltest.exe: ${IMFILL_SRCO}
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o imfilltest.exe ${IMFILL_SRCO}

clean:
	$(RM) *.o *.exe
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<

%.o: %.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ -c $<
	