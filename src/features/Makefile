
include Makefile.inc

EXEC_ENGINE_DIR=execEngine
EXAMPLES_DIR=examples

SUBDIRS	= $(EXEC_ENGINE_DIR) $(EXAMPLES_DIR) cuda

all: libfeatures.so subdirs

libfeatures.so: cuda $(OBJS)
	$(CXX) ${CXX_FLAGS} ${OPENCV_INCLUDE} -shared -o libfeatures.so ${OBJS} ${OPENCV_LIBS_DIR} ${OPENCV_LIBS} ${OPENCV_LIBS_GPU} ${CUDA_LIBS}

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. ${OPENCV_INCLUDE} -c $<

cuda: force_look
	cd cuda; $(MAKE)	

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<

subdirs: force_look
	-for d in $(SUBDIRS); do (cd $$d; $(MAKE) ); done

force_look:
	true

clean:
	rm -rf *.o libfeatures.so
	-for d in $(SUBDIRS); do (cd $$d; $(MAKE) clean ); done
