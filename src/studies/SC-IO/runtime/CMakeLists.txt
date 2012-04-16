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
#target_link_libraries("${libname}" ${OpenCV_LIBS})



#### ADIOS code
#set(adios_gpulibsrcs SCIOUtilsADIOS.cpp)
#set(adios_libname scio-adios)
#if (ADIOS_FOUND)
## Add rule to generate execution engine, as a shared library (.so)
#	if(CUDA_FOUND)
#		cuda_add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
#		set_target_properties(${adios_libname} PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_ADIOS -fPIC") 
#	else(CUDA_FOUND)
#		add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
#		set_target_properties(${adios_libname} PROPERTIES COMPILE_FLAGS "-DHAVE_ADIOS -fPIC")
#	endif(CUDA_FOUND)
#	target_link_libraries(${adios_libname} segment ${ADIOS_LIBRARIES} ${OpenCV_LIBS})
#endif(ADIOS_FOUND)




set(programs test-schedule)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} scio-runtime )


	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND)
endforeach(program)


