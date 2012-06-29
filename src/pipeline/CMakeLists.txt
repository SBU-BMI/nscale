project( Pipeline )

# The version number
set(Pipeline_VERSION_MAJOR 1)
set(Pipeline_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Segmentation_SOURCE_DIR})
include_directories(${FeaturesComputation_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Segmentation_SOURCE_DIR})
link_directories(${FeaturesComputation_SOURCE_DIR})


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")

#### for Keeneland, and possibly Jaguar
#### the default CMake module SHOULD work, if a hint of HDF5_ROOT is provided in the environment
#### this is where bin/h5cc is.  Note that on Keeneland, there is a /usr/bin/h5cc which is not right.
####  also note that HDF5 is not compiled with CMake on keeneland, so there is not HDF5-config.cmake file for the 
####  FindHDF5.cmake file in the HDF5 source directory to work against.
set(ENV{HDF5_ROOT} $ENV{HDF5_DIR})
# has no downstream effect... set(HDF5_DIR $ENV{HDF5_DIR})

#### HDF5
find_package(HDF5 REQUIRED)
if (HDF5_FOUND)
#	message(STATUS "HDF5_DEF = ${HDF5_DEFINITIONS}")
#	message(STATUS "HDF5LIBRARY_DIRS flags: ${HDF5_LIBRARY_DIRS}")
#	message(STATUS "HDF5INCLUDE_DIRS include: ${HDF5_INCLUDE_DIRS}")
#	message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}")
#	message(STATUS "HDF5 HL libs: ${HDF5_HL_LIBRARIES}")
#	message(STATUS "HDF5C libs: ${HDF5_C_LIBRARIES}")
#	message(STATUS "HDF5CXX libs: ${HDF5_CXX_LIBRARIES}")        
	
	include_directories(${HDF5_INCLUDE_DIRS})
	add_definitions(${HDF5_DEFINITIONS})
	link_directories(${HDF5_LIBRARY_DIRS})

# HDF5_LIBRARIES as defined on Keeneland is broken.  it does not respect the HDF5_DIR.
# besides there should not be absolute paths here anyways.
	set(HDF5_LIBRARIES "hdf5_hl;hdf5;z;m")
	message(STATUS "Redefined HDF5 libs: ${HDF5_LIBRARIES}")
        
endif(HDF5_FOUND)




# Source files used by pipeline
set(libsrcs h5utils.cpp )

set(libname pipeline)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs} )
if(HDF5_FOUND)
	target_link_libraries(${libname} ${HDF5_LIBRARIES})
endif(HDF5_FOUND) 


set(fprograms nu-features)
foreach(program ${fprograms})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	
	target_link_libraries(${program} features)
	target_link_libraries(${program} segment)
	target_link_libraries(${program} pipeline)
	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 
	
	if(HDF5_FOUND)
		target_link_libraries(${program} ${HDF5_LIBRARIES})
	endif(HDF5_FOUND) 
	
endforeach(program)

set(programs nu-features-by-image nu-features-file-upgrade nu-features-sample nu-features-summarize)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
		target_link_libraries(${program} segment)
		target_link_libraries(${program} pipeline)
		
	
	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 

	if(HDF5_FOUND)
		target_link_libraries(${program} ${HDF5_LIBRARIES})
	endif(HDF5_FOUND) 
	
endforeach(program)

#set(kprograms nu-features-kmeans)
#foreach(program ${kprograms})
#	# Add an executable to be built from the files listed
#	add_executable(${program} ${program}.cpp)
#
#	# Link the executable to the Features Computation library
#		target_link_libraries(${program} segment)
#		target_link_libraries(${program} pipeline)
#		target_link_libraries(${program} kmeans)
#	
#	if(MPI_FOUND)
#		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
#  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
#  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
#		target_link_libraries(${program} ${MPI_LIBRARIES})
#	endif(MPI_FOUND) 
#
#	if(HDF5_FOUND)
#		target_link_libraries(${program} ${HDF5_LIBRARIES})
#	endif(HDF5_FOUND) 
#	
#endforeach(program)


