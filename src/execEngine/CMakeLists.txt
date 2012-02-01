cmake_minimum_required(VERSION 2.6)
project(ExecEngine)

set(libname execengine)

# The version number
set(ExecEngine_VERSION_MAJOR 1)
set(ExecEngine_VERSION_MINOR 1)

# Source files used by Execution Engine
set(libsrcs Task.cpp TasksQueue.cpp ExecutionEngine.cpp ThreadPool.cpp)

# Find threads package which is linked to the execution engine library
find_package( Threads REQUIRED )

# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})

# Link Pthreads stuff to the execution engine
target_link_libraries("${libname}" ${CMAKE_THREAD_LIBS_INIT})

# Check if the user want to build sample applications
option (BUILD_SAMPLE_APPLICTIONS "Inform whether sample applications should be built" ON)

# if build same applications is set, than build them: currently just one
if(BUILD_SAMPLE_APPLICTIONS)
	add_subdirectory("examples/taskHello/")
endif(BUILD_SAMPLE_APPLICTIONS)
