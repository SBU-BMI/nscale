project( Runtime )

# The version number
set(Runtime_VERSION_MAJOR 1)
set(Runtime_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})
include_directories(${SCIOSegmentation_SOURCE_DIR})


# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Runtime_BINARY_DIR})
link_directories(${SCIOSegmentation_BINARY_DIR})

#### Runtime code

# Source files used by Runtime
set(libsrcs Debug.cpp Communicator_I.cpp CommHandler_I.cpp Process.cpp PullCommHandler.cpp PushCommHandler.cpp RandomScheduler.cpp RoundRobinScheduler.cpp)

set(libname runtime)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})
# Link opencv stuff to the execution engine

target_link_libraries(${libname} nu-io)

if(MPI_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
	set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
	target_link_libraries(${libname} ${MPI_LIBRARIES})
endif(MPI_FOUND)


#add_subdirectory(test/src)
add_subdirectory(example/test)
add_subdirectory(example/ADIOS)


