project( SCIOSegmentation )

# The version number
set(SCIOSegmentation_VERSION_MAJOR 1)
set(SCIOSegmentation_VERSION_MINOR 0)



# Make sure the compiler can find the includes
include_directories(${Segmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Segmentation_SOURCE_DIR})


# Source files used by Segmentation
set(libsrcs SCIOUtilsLogger.cpp SCIOHistologicalEntities.cpp )
set(gpulibsrcs SCIOUtilsCVImageIO.cpp SCIOHistologicalEntitiesGPU.cpp)

#include_directories(${Segmentation_SOURCE_DIR}/cuda)

set(libname scio-segment)

# Add rule to generate execution engine, as a shared library (.so)
if(CUDA_FOUND)
	cuda_add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS -DHAVE_CUDA) 
else(CUDA_FOUND)
	add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)

# Link opencv stuff to the execution engine
target_link_libraries("${libname}" segment ${OpenCV_LIBS})

set(programs nu-segment-scio)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} scio-segment segment ${OpenCV_LIBS})
	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 
	
endforeach(program)

