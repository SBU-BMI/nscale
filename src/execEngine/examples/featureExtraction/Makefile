
include ../../Makefile.inc

all: featuresExtraction 

FEATURES_DIR=../../../features
EXEC_ENGINE_DIR=../../


%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I. -I${EXEC_ENGINE_DIR} -I${FEATURES_DIR} ${OPENCV_INCLUDE} -c $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<

featuresExtraction: featuresExtraction.cpp featuresExtraction.o TaskFeature.o
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I. -I${EXEC_ENGINE_DIR} -I${FEATURES_DIR} featuresExtraction.cpp TaskFeature.o -o featuresExtraction ${OPENCV_LIBS_DIR}  ${OPENCV_LIBS} $(CUDA_LIBS) -L$(FEATURES_DIR) -lfeatures -L${EXEC_ENGINE_DIR} -lexecengine -fopenmp



clean:
	rm -rf featuresExtraction *.o *.dSYM 
