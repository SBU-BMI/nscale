
include ../Makefile.inc

all: compFeatures

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. ${OPENCV_INCLUDE} -c $<

compFeatures: CompFeatures.cpp
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. CompFeatures.cpp -o compFeatures ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} -L. -lfeatures

clean:
	rm -rf compFeatures *.dSYM 
