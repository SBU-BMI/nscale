
include ../Makefile.inc

CXX_FLAGS+=-DPRINT_FEATURES

all: compFeatures compFeaturesPipeline compFeaturesPipelineHEG

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. ${OPENCV_INCLUDE} -c $<

compFeatures: CompFeatures.cpp
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. CompFeatures.cpp -o compFeatures ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} -L.. -lfeatures

compFeaturesPipeline: CompFeaturesPipeline.cpp
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. CompFeaturesPipeline.cpp -o compFeaturesPipeline ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} -L.. -lfeatures

compFeaturesPipelineHEG: CompFeaturesPipelineHEG.cpp ColorDeconv_final.o BGR2GRAY.o
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I.. CompFeaturesPipelineHEG.cpp ColorDeconv_final.o BGR2GRAY.o -o compFeaturesPipelineHEG ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} -L.. -lfeatures


clean:
	rm -rf compFeatures compFeaturesPipeline compFeaturesPipelineHEG *.o *.dSYM 
