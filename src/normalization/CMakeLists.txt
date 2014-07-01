project( Normalization )

# The version number
set(Normalization_VERSION_MAJOR 1)
set(Normalization_VERSION_MINOR 0)


# Source files used by Segmentation
set(libsrcs Normalization.cpp)
#set(gpulibsrcs ObjFeaturesGPU.cpp)
#set(cudalibsrcs cuda/hist-ops.cu cuda/gradient.cu cuda/features.cu)

include_directories(${Normalization_SOURCE_DIR})
#include_directories(${FeaturesAll_SOURCE_DIR}/cuda)
include_directories(${Segmentation_SOURCE_DIR})
include_directories(${cci_common_SOURCE_DIR})

set(libname normalization)


if(CUDA_FOUND)
	cuda_add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs} ${cudalibsrcs})
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_CUDA") 
else(CUDA_FOUND)
	add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)

# Link opencv stuff to opencv
target_link_libraries("${libname}" ${OpenCV_LIBS})
target_link_libraries("${libname}" segment)

# if build test applications is set, than build them: currently just one
if(BUILD_TEST_APPLICATIONS)
	add_subdirectory("examples/")
endif(BUILD_TEST_APPLICATIONS)

