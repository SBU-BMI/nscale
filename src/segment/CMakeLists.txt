cmake_minimum_required(VERSION 2.6)
project( Segmentation )

# The version number
set(Segmentation_VERSION_MAJOR 1)
set(Segmentation_VERSION_MINOR 0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)



# Source files used by Execution Engine
set(libsrcs HistologicalEntities.cpp MorphologicOperations.cpp PixelOperations.cpp NeighborOperations.cpp FileUtils.cpp )
set(gpulibsrcs HistologicalEntitiesGPU.cpp MorphologicOperationsGPU.cpp PixelOperationsGPU.cpp NeighborOperationsGPU.cpp UtilsCVImageIO.cpp)
set(cudalibsrcs cuda/global_queue.cu cuda/neighbor-ops.cu cuda/watershed-ca-korbes.cu cuda/watershed-dw-korbes.cu cuda/queue.cu cuda/imrecon_queue_int_kernel.cu cuda/pixel-ops.cu cuda/change_kernel.cu cuda/imreconstruct_binary_kernel.cu cuda/imreconstruct_int_kernel.cu cuda/imreconstruct_float_kernel.cu )

include_directories(${Segmentation_SOURCE_DIR}/cuda)

set(libname segment)
set(gpulibname gpusegment)

#### OpenCV
find_package( OpenCV REQUIRED )

#### CUDA
set(WITH_CUDA ON CACHE BOOL "Include NVidia Cuda Runtime support")
if(WITH_CUDA)
    find_package(CUDA 4.0)
    
    if(CUDA_FOUND) 		
        message(STATUS "CUDA detected: " ${CUDA_VERSION})

        # These vars will be processed in other scripts
     	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math -arch=sm_20 -DPREFIX_SUM)
     	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -G3)
        message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")
		add_definitions( -DWITH_CUDA )
    endif(CUDA_FOUND)
else(WITH_CUDA)
	set(CUDA_FOUND 0)
endif(WITH_CUDA)


#### MPI
set(WITH_MPI OFF CACHE BOOL "Build with MPI support")
if (WITH_MPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
else(WITH_MPI)
	set(MPI_FOUND 0)
endif(WITH_MPI)

#### OpenMP
set(WITH_OPENMP OFF CACHE BOOL "Build with OpenMP support")
if (WITH_OPENMP)
    include(FindOpenMP)
    
	 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(WITH_OPENMP)



# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs} )
if(CUDA_FOUND)
	cuda_add_library("${gpulibname}" SHARED ${gpulibsrcs} ${cudalibsrcs})
	set_target_properties(${gpulibname} PROPERTIES COMPILE_FLAGS -DHAVE_CUDA) 
else(CUDA_FOUND)
	add_library("${gpulibname}" SHARED ${gpulibsrcs} )
endif(CUDA_FOUND)

# Link opencv stuff to the execution engine
target_link_libraries("${libname}" ${OpenCV_LIBS})
target_link_libraries("${gpulibname}" ${OpenCV_LIBS})


set(CMAKE_EXECUTABLE_SUFFIX ".exe")


set(programs nu-segment)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} segment gpusegment)
	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 
	
endforeach(program)



# Check if the user want to build test applications
option (BUILD_TEST_APPS "Inform whether test applications should be built" ON)

# if build test applications is set, than build them: currently just one
if(BUILD_TEST_APPS)
	add_subdirectory("test/src")
endif(BUILD_TEST_APPS)


