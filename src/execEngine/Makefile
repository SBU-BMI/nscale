include Makefile.inc

EXEC_ENGINE_OBJS=Task.o TasksQueue.o ExecutionEngine.o ThreadPool.o warmUp.o

SUBDIRS=tests #examples/featureExtraction/

all: libexecengine.so subdirs

libexecengine.so: $(EXEC_ENGINE_OBJS)
	$(CXX) ${CXX_FLAGS} -shared -o libexecengine.so ${EXEC_ENGINE_OBJS} -lpthread ${EXEC_ENGINE_LIBS} 

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. -I${OPENCV_INCLUDE} -c $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(NVCC_INCLUDE) -c $<

subdirs: force_look
	-for d in $(SUBDIRS); do (cd $$d; $(MAKE) ); done

force_look:
	true

clean:
	rm -rf *.o libexecengine.so 
	-for d in $(SUBDIRS); do (cd $$d; $(MAKE) clean ); done
