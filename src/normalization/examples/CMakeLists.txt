cmake_minimum_required(VERSION 2.6)
project(NormalizationTest)

# Make sure the compiler can find the includes in the features directory
include_directories(${Normalization_SOURCE_DIR})

# Make sure the compiler can find the library in features directory
link_directories(${Normalization_BINARY_DIR})

include_directories(${cci_common_SOURCE_DIR})
link_directories(${cci_common_BINARY_DIR})


set(CMAKE_EXECUTABLE_SUFFIX ".exe")

set(programs normTest)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
	target_link_libraries(${program} normalization cci_common)
endforeach(program)


