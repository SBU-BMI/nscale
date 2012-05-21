project( SCIORuntime )

# The version number
set(SCIORuntime_VERSION_MAJOR 1)
set(SCIORuntime_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Runtime_SOURCE_DIR})


#### Runtime code

# Source files used by Runtime
set(libsrcs Process.cpp Work.cpp UniformRandomSchedule.cpp RoundRobinSchedule.cpp FIFOSchedule.cpp Parameters.cpp CoordinatedWork.cpp LimitedParamQueue.cpp PrioritizedParamQueue.cpp )

set(libname scio-runtime)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})
# Link opencv stuff to the execution engine


set(programs test-schedule)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} scio-runtime )


	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND)
endforeach(program)


