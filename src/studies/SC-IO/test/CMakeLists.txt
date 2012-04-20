cmake_minimum_required(VERSION 2.6)
project(SCIOTest)

# Make sure the compiler can find the includes in the segmentation directory
#include_directories(${SCIOSegmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Segmentation directory
#link_directories(${SCIOSegmentation_SOURCE_DIR})

set(CMAKE_EXECUTABLE_SUFFIX ".exe")

	if(MPI_FOUND)
set(programs test-mpi)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
endforeach(program)

	endif(MPI_FOUND)

	if(ADIOS_FOUND)
set(programs test-adios)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI -DWITH_ADIOS")
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} scio-adios ${ADIOS_LIBRARIES} ${MPI_LIBRARIES})

endforeach(program)

### copy over the adios xml files
FILE(GLOB ADIOS_XMLS RELATIVE ${PROJECT_SOURCE_DIR} "*.xml" )
set(ADIOS_XML_DIR ${CMAKE_BINARY_DIR}/adios_xml)
#message("adios xml dir = ${ADIOS_XML_DIR}")
make_directory(${ADIOS_XML_DIR})
foreach(ADIOS_XML ${ADIOS_XMLS})
#	message("copy from ${PROJECT_SOURCE_DIR}/${ADIOS_XML} to ${ADIOS_XML_DIR}/${ADIOS_XML}")
	configure_file(	${PROJECT_SOURCE_DIR}/${ADIOS_XML} ${ADIOS_XML_DIR}/${ADIOS_XML} COPY_ONLY)
endforeach(ADIOS_XML)


	endif(ADIOS_FOUND)

	
	