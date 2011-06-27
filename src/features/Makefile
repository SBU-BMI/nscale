CXX_FLAGS = -g

all: Contour.o Blob.o RegionalMorphologyAnalysis.o testRegional testContour testBlob features 
	
	
features: features.cpp
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -o features features.cpp

Contour.o: Contour.h Contour.cpp
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -c Contour.cpp
	
Blob.o: Blob.h Blob.cpp
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -c Blob.cpp
	
RegionalMorphologyAnalysis.o: RegionalMorphologyAnalysis.h RegionalMorphologyAnalysis.cpp
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -c RegionalMorphologyAnalysis.cpp
	
testContour: Test.cpp Contour.o
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -o testContour Test.cpp Contour.o

testBlob: TestBlob.cpp Contour.o Blob.o
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -o testBlob TestBlob.cpp Contour.o Blob.o

testRegional: TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o
	g++ `pkg-config --cflags opencv --libs opencv` ${CXX_FLAGS} -lstdc++  -o testRegional TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o

clean:
	rm -rf features *.o testContour testBlob testRegional
