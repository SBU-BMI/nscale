CXX_FLAGS = -O3 #-g -DVISUAL_DEBUG
OPENCV_INCLUDE=/usr/local/include/opencv/
OPENCV_LIB=-lopencv_core -lopencv_highgui -lopencv_imgproc

all: Contour.o Blob.o RegionalMorphologyAnalysis.o testRegional testContour testBlob features 

#`pkg-config --cflags opencv2 --libs opencv2`
	
	
features: features.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -o features features.cpp

Contour.o: Contour.h Contour.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c Contour.cpp
	
Blob.o: Blob.h Blob.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c Blob.cpp

DrawAuxiliar.o: DrawAuxiliar.h DrawAuxiliar.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c DrawAuxiliar.cpp

RegionalMorphologyAnalysis.o: RegionalMorphologyAnalysis.h RegionalMorphologyAnalysis.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c RegionalMorphologyAnalysis.cpp
	
testContour: Test.cpp Contour.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -o testContour Test.cpp Contour.o ${OPENCV_LIB} 

testBlob: TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -o testBlob TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o ${OPENCV_LIB}

testRegional: TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -o testRegional TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o ${OPENCV_LIB} 

clean:
	rm -rf features *.o testContour testBlob testRegional
