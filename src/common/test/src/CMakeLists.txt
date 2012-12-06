project(cci_common_test)

cmake_minimum_required(VERSION 2.6)

# Make sure the compiler can find the includes in the segmentation directory
include_directories(${cci_common_SOURCE_DIR})

# Make sure the compiler can find the library in Segmentation directory
link_directories(${cci_common_BINARY_DIR})

set(programs FileUtilsTest)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
	target_link_libraries(${program} cci_common)
endforeach(program)


