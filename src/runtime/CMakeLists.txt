project( Runtime )

# The version number
set(Runtime_VERSION_MAJOR 1)
set(Runtime_VERSION_MINOR 0)


if(NOT MPI_FOUND)
    message(SEND_ERROR "MPI was not found.  libruntime requires it" )
endif(NOT MPI_FOUND)



#### CCI_COMMON library
# Make sure the compiler can find the includes
include_directories(${cci_common_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${cci_common_BINARY_DIR})

#### Runtime library
# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})


#### Runtime code
# Source files used by Runtime
set(libsrcs CmdlineParser.cpp Communicator_I.cpp CommHandler_I.cpp Process.cpp PullCommHandler.cpp PushCommHandler.cpp RandomScheduler.cpp RoundRobinScheduler.cpp NullSinkAction.cpp DataBuffer.cpp MPIDataBuffer.cpp MPISendDataBuffer.cpp MPIRecvDataBuffer.cpp)

set(libname runtime)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})
# Link opencv stuff to the execution engine

target_link_libraries(${libname} cci_common)
target_link_libraries(${libname} ${Boost_LIBRARIES})

set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
target_link_libraries(${libname} ${MPI_LIBRARIES})
	
	
#### testing code
if (BUILD_TEST_APPLICATIONS)
    add_subdirectory(test)
endif(BUILD_TEST_APPLICATIONS)
