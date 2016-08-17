cmake_minimum_required(VERSION 2.6)
project(FeaturesAllTest)

# Make sure the compiler can find the includes in the features directory
include_directories(${FeaturesAll_SOURCE_DIR})

# Make sure the compiler can find the library in features directory
link_directories(${FeaturesAll_BINARY_DIR})
include_directories(${Segmentation_SOURCE_DIR})

include_directories(${cci_common_SOURCE_DIR})
link_directories(${cci_common_BINARY_DIR})


set(CMAKE_EXECUTABLE_SUFFIX ".exe")

set(programs featuresTest featuresTest2 mainFeatures)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
	target_link_libraries(${program} segment featuresAll cci_common)
endforeach(program)


