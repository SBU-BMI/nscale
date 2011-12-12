
include ../../../Makefile.inc

all: featuresExtraction 

LIB_FEATURES_DIR=-L../../../
LIB_EXEC_ENGINE_DIR=-L../../


%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I. -I../.. -I../../.. ${OPENCV_INCLUDE} -c $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<

featuresExtraction: featuresExtraction.cpp featuresExtraction.o TaskFeature.o
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I. -I../.. -I../../.. featuresExtraction.cpp TaskFeature.o -o featuresExtraction ${OPENCV_LIBS_DIR}  ${OPENCV_LIBS} $(CUDA_LIBS) $(LIB_FEATURES_DIR) -lfeatures $(LIB_EXEC_ENGINE_DIR) -lexecengine -fopenmp



clean:
	rm -rf featuresExtraction *.o *.dSYM 
