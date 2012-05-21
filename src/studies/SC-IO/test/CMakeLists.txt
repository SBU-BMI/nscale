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
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
	  	set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endforeach(program)
endif(MPI_FOUND)

if(ADIOS_FOUND)

	
	### copy over the adios xml files
	FILE(GLOB ADIOS_XMLS RELATIVE ${PROJECT_SOURCE_DIR} "*.xml" )
	set(ADIOS_XML_DIR ${CMAKE_BINARY_DIR}/adios_xml)
	#message("adios xml dir = ${ADIOS_XML_DIR}")
	make_directory(${ADIOS_XML_DIR})
	foreach(ADIOS_XML ${ADIOS_XMLS})
	#	message("copy from ${PROJECT_SOURCE_DIR}/${ADIOS_XML} to ${ADIOS_XML_DIR}/${ADIOS_XML}")
		configure_file(	${PROJECT_SOURCE_DIR}/${ADIOS_XML} ${ADIOS_XML_DIR}/${ADIOS_XML} COPY_ONLY)
		
		  execute_process(
	    COMMAND python ${ADIOS_DIR}/bin/gpp.py ${PROJECT_SOURCE_DIR}/${ADIOS_XML}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	  ) 
		
	endforeach(ADIOS_XML)
	FILE(GLOB ADIOS_CHS RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "*.ch" )
	
	include_directories(${PROJECT_SOURCE_DIR}/../adios-include)
#	message(STATUS "TEST ADIOS INCLUDES = ${ADIOS_COMP_FLAGS}")
	
	set(TARGET_DEFS "WITH_ADIOS")
	set(TARGET_COMP_FLAGS "${ADIOS_COMP_FLAGS}")
	set(TARGET_LINK_FLAGS "")
	if (MPI_FOUND)	
		set(TARGET_DEFS "${TARGET_DEFS};WITH_MPI")
		set(TARGET_COMP_FLAGS "${TARGET_COMP_FLAGS} ${MPI_COMPILE_FLAGS}")
		set(TARGET_LINK_FLAGS "${TARGET_LINK_FLAGS} ${MPI_LINK_FLAGS}")			
	endif(MPI_FOUND)
#	message(STATUS "TEST PROG DEFS = ${TARGET_DEFS}")
#	message(STATUS "TEST PROG COMPILE_FLAGS = ${TARGET_COMP_FLAGS}")
#	message(STATUS "TEST PROG LINK_FLAGS = ${TARGET_LINK_FLAGS}")
#	message(STATUS "TEST PROG ADIOS LIB DIRS = ${ADIOS_LIB_DIRS}")
#	message(STATUS "TEST PROG ADIOS LIBS = ${ADIOS_LIBS}")
	set(programs test-adios)
	foreach(program ${programs})
		# Add an executable to be built from the files listed
		add_executable(${program} ${program}.cpp)
	
		# Link the executable to the Segmentation library

		target_link_libraries(${program} ${adios_libname} ${io_libname} ${ADIOS_LIBS})
		if (MPI_FOUND)
			target_link_libraries(${program} ${MPI_LIBRARIES})
		endif(MPI_FOUND)
		include_directories(${CMAKE_CURRENT_BINARY_DIR})
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "${TARGET_DEFS}")
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${TARGET_COMP_FLAGS}")
		set_target_properties(${program} PROPERTIES LINK_FLAGS "${TARGET_LINK_FLAGS}")
		
	endforeach(program)


endif(ADIOS_FOUND)

	
	
