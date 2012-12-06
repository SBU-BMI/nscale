project(SCIOTest)

# Make sure the compiler can find the includes in the segmentation directory
include_directories(${SCIOSegmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Segmentation directory
link_directories(${SCIOSegmentation_BINARY_DIR})

set(programs test-cpp)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Segmentation library
endforeach(program)


if(MPI_FOUND)
	set(programs test-mpi)
	foreach(program ${programs})
		# Add an executable to be built from the files listed
		add_executable(${program} ${program}.cpp)
	
		# Link the executable to the Segmentation library
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
	  	set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} nu-io ${MPI_LIBRARIES})
	endforeach(program)
endif(MPI_FOUND)

if(ADIOS_FOUND)

	
	### copy over the adios xml files
	FILE(GLOB ADIOS_XMLS_IN RELATIVE ${PROJECT_SOURCE_DIR} "*.xml.in" )
	set(ADIOS_XML_DIR ${EXECUTABLE_OUTPUT_PATH}/../adios_xml)
	#message("adios xml dir = ${ADIOS_XML_DIR}")
	make_directory(${ADIOS_XML_DIR})
	
	# create a list of suffixes.
	set(ADIOS_TRANSPORTS NULL POSIX MPI_LUSTRE MPI_AMR MPI)
	set(NULL_TRANSPORT_OPTION "")
	set(POSIX_TRANSPORT_OPTION "")
	set(MPI_TRANSPORT_OPTION "")
	set(MPI_LUSTRE_TRANSPORT_OPTION "block_size=68157440")
	set(MPI_AMR_TRANSPORT_OPTION "num_aggregator=16;num_ost=4")
	

	foreach(ADIOS_XML_IN ${ADIOS_XMLS_IN})
		# generate the xml files for different transports
		foreach (ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})

		    set(TRANSPORT_OPTIONS ${${ADIOS_TRANSPORT}_TRANSPORT_OPTION})
    		#message(STATUS "${${ADIOS_TRANSPORT}_TRANSPORT_OPTION} : ${TRANSPORT_OPTIONS}")
			string(REPLACE ".xml.in" "-${ADIOS_TRANSPORT}.xml" ADIOS_XML ${ADIOS_XML_IN})
			# then replace the transport variables.
			#message(STATUS "copy from ${PROJECT_SOURCE_DIR}/${ADIOS_XML_IN} to ${ADIOS_XML_DIR}/${ADIOS_XML}")
			configure_file(	${PROJECT_SOURCE_DIR}/${ADIOS_XML_IN} ${ADIOS_XML_DIR}/${ADIOS_XML} @ONLY)	
		endforeach(ADIOS_TRANSPORT)

	  # then use the last defined ADIOS_XML with the python script to generate the c include files.
	  execute_process(
	    COMMAND python ${ADIOS_DIR}/bin/gpp.py ${ADIOS_XML_DIR}/${ADIOS_XML}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	  )

	endforeach(ADIOS_XML_IN)



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

	
	
