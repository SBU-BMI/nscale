CXX=g++
CXX_FLAGS = -O3 -fopenmp #-g -DVISUAL_DEBUG
OBJS=Contour.o Blob.o RegionalMorphologyAnalysis.o
OPENCV_INCLUDE=/usr/local/include/opencv/
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OPENCV_LIBS_GPU=${OPENCV_LIBS} -lopencv_gpu
EXAMPLES_DIR=./examples

all: libfeatures.so examples

examples: features testContour testBlob testRegional gpu


#`pkg-config --cflags opencv2 --libs opencv2`
libfeatures.so: $(OBJS)
	$(CXX) ${CXX_FLAGS} -I${OPENCV_INCLUDE} -shared -o libfeatures.so ${OBJS} ${OPENCV_LIBS}

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c $<
	
features: ${EXAMPLES_DIR}/features.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/features.cpp -o ${EXAMPLES_DIR}/features  ${OPENCV_LIBS}

testContour: ${EXAMPLES_DIR}/Test.cpp Contour.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/Test.cpp Contour.o -o ${EXAMPLES_DIR}/testContour  ${OPENCV_LIBS}

testBlob: ${EXAMPLES_DIR}/TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o -o ${EXAMPLES_DIR}/testBlob ${OPENCV_LIBS}

testRegional: ${EXAMPLES_DIR}/TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/TestRegional.cpp RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o -o ${EXAMPLES_DIR}/testRegional ${OPENCV_LIBS} 

gpu: ${EXAMPLES_DIR}/gpu.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/gpu.cpp -o ${EXAMPLES_DIR}/gpu  ${OPENCV_LIBS_GPU}

clean:
	rm -rf  ${EXAMPLES_DIR}/gpu ${EXAMPLES_DIR}/features ${EXAMPLES_DIR}/testRegional ${EXAMPLES_DIR}/testBlob ${EXAMPLES_DIR}/testContour *.o libfeatures.so
