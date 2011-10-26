
include ../Makefile.inc
LIB_OBJS= ../cooc.o ../cudaToCInterface.o ../RegionalMorphologyAnalysis.o ../Contour.o ../Blob.o ../DrawAuxiliar.o ../Operators.o 

all: features testContour testRegional gpu #testBlob

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. ${OPENCV_INCLUDE} -c $<

# move lines bellow to examples folder makefile	
features:features.cpp
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I. features.cpp -o features ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} $(CUDA_LIBS)

testContour: Test.cpp ../Contour.o
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. Test.cpp ../Contour.o -o testContour ${OPENCV_LIBS_DIR} ${OPENCV_LIBS}

#testBlob: ../Contour.o ../Blob.o ../DrawAuxiliar.o ../Operators.o
#	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. TestBlob.cpp ../Contour.o ../Blob.o ../DrawAuxiliar.o  ../Operators.o -o testBlob ${OPENCV_LIBS}  ${OPENCV_LIBS_GPU} $(CUDA_LIBS) 

testRegional: TestRegional.cpp $(LIB_OBJS)
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. TestRegional.cpp $(LIB_OBJS) -o testRegional ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} ${OPENCV_LIBS_GPU} $(CUDA_LIBS) 

gpu: gpu.cpp
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. gpu.cpp -o gpu ${OPENCV_LIBS_DIR} ${OPENCV_LIBS_GPU}

clean:
	rm -rf features testRegional testBlob testBlobGPU gpu testContour *.dSYM 
