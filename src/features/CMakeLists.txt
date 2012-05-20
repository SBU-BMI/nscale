project( FeaturesAll )

# The version number
set(FeaturesAll_VERSION_MAJOR 1)
set(FeaturesAll_VERSION_MINOR 0)


# Source files used by Segmentation
set(libsrcs ObjFeatures.cpp Operators.cpp CytoplasmCalc.cpp)
set(gpulibsrcs ObjFeaturesGPU.cpp)
set(cudalibsrcs cuda/hist-ops.cu cuda/gradient.cu)

include_directories(${FeaturesAll_SOURCE_DIR})
include_directories(${FeaturesAll_SOURCE_DIR}/cuda)
include_directories(${Segmentation_SOURCE_DIR})

set(libname featuresAll)
if(CUDA_FOUND)
	cuda_add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs} ${cudalibsrcs})
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_CUDA") 
else(CUDA_FOUND)
	add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)

# Link opencv stuff to opencv
target_link_libraries("${libname}" ${OpenCV_LIBS})

# if build test applications is set, than build them: currently just one
if(BUILD_TEST_APPLICATIONS)
	add_subdirectory("examples/")
endif(BUILD_TEST_APPLICATIONS)

