CXX=g++
CXX_FLAGS = -fPIC -g # -fopenmp #-g -DVISUAL_DEBUG -O3 
OBJS=cooc.o cudaToCInterface.o Contour.o Blob.o RegionalMorphologyAnalysis.o DrawAuxiliar.o Operators.o
OPENCV_INCLUDE=/usr/local/include/opencv/
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OPENCV_LIBS_GPU=${OPENCV_LIBS} -lopencv_gpu
EXAMPLES_DIR=./examples

NVCC=nvcc
NVCC_FLAGS=-m64 #-g -G
NVCC_INCLUDE=-I../NVIDIA\ GPU\ Computing\ SDK/C/common/inc/ -I/usr/local/cuda/include
CUDA_LIBS= -L/usr/local/cuda/lib -lcudart


all: libfeatures.so examples

examples: features testContour testBlob testRegional gpu


#`pkg-config --cflags opencv2 --libs opencv2`
libfeatures.so: $(OBJS)
	$(CXX) ${CXX_FLAGS} -I${OPENCV_INCLUDE} -shared -o libfeatures.so ${OBJS} ${OPENCV_LIBS} ${OPENCV_LIBS_GPU} ${CUDA_LIBS}

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I${OPENCV_INCLUDE} -c $<
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<
	
features: ${EXAMPLES_DIR}/features.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/features.cpp -o ${EXAMPLES_DIR}/features  ${OPENCV_LIBS} $(CUDA_LIBS)

testContour: ${EXAMPLES_DIR}/Test.cpp Contour.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/Test.cpp Contour.o -o ${EXAMPLES_DIR}/testContour  ${OPENCV_LIBS}

testBlob: ${EXAMPLES_DIR}/TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/TestBlob.cpp Contour.o Blob.o DrawAuxiliar.o -o ${EXAMPLES_DIR}/testBlob ${OPENCV_LIBS}

testRegional: ${EXAMPLES_DIR}/TestRegional.cpp cooc.o cudaToCInterface.o RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o Operators.o
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/TestRegional.cpp cooc.o cudaToCInterface.o RegionalMorphologyAnalysis.o Contour.o Blob.o DrawAuxiliar.o Operators.o -o ${EXAMPLES_DIR}/testRegional ${OPENCV_LIBS} ${OPENCV_LIBS_GPU} $(CUDA_LIBS) 

gpu: ${EXAMPLES_DIR}/gpu.cpp
	g++ ${CXX_FLAGS} -I${OPENCV_INCLUDE} -I. ${EXAMPLES_DIR}/gpu.cpp -o ${EXAMPLES_DIR}/gpu  ${OPENCV_LIBS_GPU}

clean:
	rm -rf ${EXAMPLES_DIR}/features ${EXAMPLES_DIR}/testRegional ${EXAMPLES_DIR}/testBlob ${EXAMPLES_DIR}/testBlobGPU ${EXAMPLES_DIR}/gpu ${EXAMPLES_DIR}/testContour *.o libfeatures.so
