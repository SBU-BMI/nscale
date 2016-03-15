cmake_minimum_required(VERSION 2.6)
project(SegmentationTest)

# Make sure the compiler can find the includes in the segmentation directory
include_directories(${Segmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Segmentation directory
link_directories(${Segmentation_BINARY_DIR})
include_directories(${cci_common_SOURCE_DIR})
link_directories(${cci_common_BINARY_DIR})
include_directories(${Normalization_SOURCE_DIR}/)


if(OPENMP_FOUND)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(programs imrecon3DTest bwareaopenTest diff deconvTest distTransform copyMakeBorder imfillTest imhminTest imreconChunkTest imreconExperiments imreconOptimizeProfiler imreconTest localMaximaTest miscTest nu-segmentChunkTest nu-segmentTest queueTest watershedTest watershedTestUint imreconMulticore tiler tileAtile)

foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
	target_link_libraries(${program} segment normalization)
	target_link_libraries(${program} cci_common)
endforeach(program)


