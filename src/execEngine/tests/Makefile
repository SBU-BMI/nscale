include ../../Makefile.inc

EXEC_ENGINE_OBJS=../Task.o ../TasksQueue.o ../ThreadPool.o ../ExecutionEngine.o  ../warmUp.o

all: testQueue testThreadPool testEngine 

%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I.. -I../../ -I${OPENCV_INCLUDE} -c $<

testQueue: testQueue.cpp $(EXEC_ENGINE_OBJS)
	$(CXX) $(CXX_FLAGS) -I../ -I../../ testQueue.cpp $(EXEC_ENGINE_OBJS) -o testQueue -lpthread ${CUDA_LIBS}

testEngine: testEngine.cpp $(EXEC_ENGINE_OBJS)
	$(CXX) $(CXX_FLAGS) -I../ -I../../ testEngine.cpp $(EXEC_ENGINE_OBJS) -o testEngine -lpthread ${CUDA_LIBS}

testThreadPool: testThreadPool.cpp $(EXEC_ENGINE_OBJS)
	$(CXX) $(CXX_FLAGS) -I../ -I../../ testThreadPool.cpp $(EXEC_ENGINE_OBJS) -o testThreadPool -lpthread ${CUDA_LIBS}

clean:
	rm -rf testQueue testThreadPool testEngine *.o *.dSYM
