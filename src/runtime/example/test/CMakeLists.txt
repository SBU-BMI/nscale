project( Runtime-test )

# The version number
set(Runtime-test_VERSION_MAJOR 1)
set(Runtime-test_VERSION_MINOR 0)



# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Runtime_SOURCE_DIR})


#### Runtime example library

# Source files used by Runtime
set(libsrcs Assign.cpp Save.cpp Segment.cpp SegConfigurator.cpp)

set(libname rt_example)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})
# Link opencv stuff to the execution engine

if(MPI_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
	set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
	target_link_libraries(${libname} ${MPI_LIBRARIES})
endif(MPI_FOUND)


#### Runtime example code

set(programs Assign_test Segment_test Save_test PullHandler_test PushHandler_test)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} runtime )
	target_link_libraries(${program} rt_example )

	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND)
	
	add_test(${program} ${program})
endforeach(program)


