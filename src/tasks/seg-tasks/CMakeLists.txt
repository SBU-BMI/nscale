project( SegmentationTasks )

# The version number
set(SegmentationTasks_VERSION_MAJOR 1)
set(SegmentationTasks_VERSION_MINOR 0)


include_directories(${Segmentation_SOURCE_DIR})
link_directories(${Segmentation_SOURCE_DIR})
include_directories(${FeaturesComputation_SOURCE_DIR})
link_directories(${FeaturesComputation_SOURCE_DIR})
include_directories(${ExecEngine_SOURCE_DIR})
link_directories(${ExecEngine_SOURCE_DIR})

# Source files used by seg tasks
file(GLOB libsrcs "*Task.cpp")

set(libname segtasks)

# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs} )
if(CUDA_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS -DHAVE_CUDA) 
endif(CUDA_FOUND)

# Link opencv stuff to the execution engine
target_link_libraries("${libname}" ${OpenCV_LIBS})
target_link_libraries("${libname}" segment)
target_link_libraries("${libname}" features)
target_link_libraries("${libname}" execengine)


set(programs nu-segtasks-mpi-bot)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} segment)
	target_link_libraries(${program} segtasks)
	target_link_libraries(${program} features)
	target_link_libraries(${program} execengine)
	
	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 
	
endforeach(program)


