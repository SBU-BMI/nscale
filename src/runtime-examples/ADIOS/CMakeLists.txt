project( RT-Example-ADIOS )

# The version number
set(RT-Example-test_VERSION_MAJOR 1)
set(RT-Example-test_VERSION_MINOR 0)

# Make sure the compiler can find the includes
include_directories(${Runtime_SOURCE_DIR})
include_directories(${Segmentation_SOURCE_DIR})
include_directories(${SCIOSegmentation_SOURCE_DIR})
include_directories(${SCIOSegmentation_SOURCE_DIR}/adios-include)
include_directories(${SCIOSegmentation_BINARY_DIR})
include_directories(${cci_common_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Runtime_BINARY_DIR})
link_directories(${Segmentation_BINARY_DIR})
link_directories(${SCIOSegmentation_BINARY_DIR})
link_directories(${ADIOS_LIB_DIRS})
link_directories(${cci_common_BINARY_DIR})


#### Runtime example library

# Source files used by Runtime
set(libsrcs AssignTiles.cpp ADIOSSave_Reduce.cpp POSIXRawSave.cpp Segment.cpp SynthSegment.cpp SegConfigurator.cpp CVImage.cpp UtilsADIOS.cpp)

set(libname rt_adios)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})
# Link opencv stuff to the execution engine
target_link_libraries(${libname} segment)
target_link_libraries(${libname} scio-segment)
target_link_libraries(${libname} ${OpenCV_LIBS})
target_link_libraries(${libname} nu-io)
target_link_libraries(${libname} scio-adios)
target_link_libraries(${libname} ${ADIOS_LIBS})

target_link_libraries(${libname} ${ZLIB_LIBRARIES})

if(MPI_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} ${ADIOS_COMP_FLAGS}") 
	set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
	target_link_libraries(${libname} ${MPI_LIBRARIES})
else(MPI_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${ADIOS_COMP_FLAGS}")
endif(MPI_FOUND)



#### Runtime code
set(programs SegmentNuclei CVImage_test)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} runtime )
    target_link_libraries(${program} rt_adios)
    target_link_libraries(${program} segment)
    target_link_libraries(${program} ${ADIOS_LIBS})

	if(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} ${ADIOS_COMP_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	else(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${ADIOS_COMP_FLAGS}")
	
	endif(MPI_FOUND)
	
	add_test(${program} ${program})
endforeach(program)


