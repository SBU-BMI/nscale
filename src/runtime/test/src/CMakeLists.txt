project( RT-Example-test )

# The version number
set(RT-Example-test_VERSION_MAJOR 1)
set(RT-Example-test_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Runtime_SOURCE_DIR})


#### Runtime code

set(programs Process_test )
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} runtime )


	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND)
	
	add_test(${program} ${program})
endforeach(program)


