project( Pipeline )

# The version number
set(Pipeline_VERSION_MAJOR 1)
set(Pipeline_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Segmentation_SOURCE_DIR})
include_directories(${FeaturesComputation_SOURCE_DIR})
include_directories(${cci_common_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Segmentation_BINARY_DIR})
link_directories(${FeaturesComputation_BINARY_DIR})
link_directories(${cci_common_BINARY_DIR})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")

set(fprograms nu-features)
foreach(program ${fprograms})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	
	target_link_libraries(${program} features)
	target_link_libraries(${program} segment)
	target_link_libraries(${program} cci_common)
	if(HDF5_FOUND)
		target_link_libraries(${program} ${HDF5_LIBRARIES})
	endif(HDF5_FOUND) 
	
	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 
	
endforeach(program)

set(programs nu-features-by-image nu-features-file-upgrade nu-features-sample nu-features-summarize)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} cci_common)
	if(HDF5_FOUND)
		target_link_libraries(${program} ${HDF5_LIBRARIES})
	endif(HDF5_FOUND) 
	
	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND) 

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


