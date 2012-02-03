cmake_minimum_required(VERSION 2.6)
project(SegmentationTest)

# Make sure the compiler can find the includes in the segmentation directory
include_directories(${Segmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Segmentation directory
link_directories(${Segmentation_SOURCE_DIR})

set(CMAKE_EXECUTABLE_SUFFIX ".exe")

set(programs bwareaopenTest deconvTest imfillTest imreconChunkTest imreconExperiments imreconOptimizeProfiler imreconTest localMaximaTest miscTest nu-segmentChunkTest nu-segmentTest queueTest watershedTest)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
	target_link_libraries(${program} segment gpusegment)
endforeach(program)


