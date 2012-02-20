project(ExecEngine)

# The version number
set(ExecEngine_VERSION_MAJOR 1)
set(ExecEngine_VERSION_MINOR 1)

set(libname execengine)

# Source files used by Execution Engine
set(libsrcs Task.cpp TasksQueue.cpp ExecutionEngine.cpp ThreadPool.cpp)

# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})

# Link Pthreads stuff to the execution engine
target_link_libraries("${libname}" ${CMAKE_THREAD_LIBS_INIT})

# if build same applications is set, than build them: currently just one
if(BUILD_EXAMPLE_APPLICTIONS)
	add_subdirectory("examples/taskHello/")
endif(BUILD_EXAMPLE_APPLICTIONS)
