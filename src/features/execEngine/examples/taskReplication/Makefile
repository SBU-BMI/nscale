
include ../../../Makefile.inc

all: TaskReplication 

LIB_FEATURES_DIR=-L../../../
LIB_EXEC_ENGINE_DIR=-L../../


%.o: %.cpp
	$(CXX) ${CXX_FLAGS} -I. -I../.. -I../../.. ${OPENCV_INCLUDE} -c $<

TaskReplication: TaskReplication.o TaskSleep.o
	g++ ${CXX_FLAGS} ${OPENCV_INCLUDE} -I. -I../.. -I../../.. TaskSleep.o TaskReplication.o -o TaskReplication $(LIB_EXEC_ENGINE_DIR) -lexecengine 



clean:
	rm -rf TaskReplication *.o *.dSYM 
