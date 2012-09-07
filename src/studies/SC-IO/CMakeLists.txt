project( SCIOSegmentation )

# The version number
set(SCIOSegmentation_VERSION_MAJOR 1)
set(SCIOSegmentation_VERSION_MINOR 0)

#### ADIOS
set(USE_ADIOS OFF CACHE BOOL "Build with ADIOS support")
# TODO:  call adios_config at some directory, parse the output to get the ADIOS C flags and LD flags.  save and use later. 
if (USE_ADIOS)
	find_path(ADIOS_DIR include/adios.h
		HINTS /usr /usr/local /usr/local/share /opt/local ENV ADIOS_DIR
		DOC "path to adios root directory")
	message(STATUS "ADIOS dir = ${ADIOS_DIR}")

	if ( ADIOS_DIR )
	set(TEMP $ENV{ADIOS_INC}) 
	if ( NOT TEMP )
		if (MPI_FOUND)
			EXECUTE_PROCESS(COMMAND ${ADIOS_DIR}/bin/adios_config -c
				OUTPUT_VARIABLE TEMP
				OUTPUT_STRIP_TRAILING_WHITESPACE)
#			message(STATUS "ADIOS INC FLAGS = ${TEMP}")
		else(MPI_FOUND)
			EXECUTE_PROCESS(COMMAND ${ADIOS_DIR}/bin/adios_config -c -s
				OUTPUT_VARIABLE TEMP
				OUTPUT_STRIP_TRAILING_WHITESPACE)
#			message(STATUS "ADIOS INC FLAGS = ${TEMP}")
		endif(MPI_FOUND)
	endif(NOT TEMP)
	set(ADIOS_COMP_FLAGS ${TEMP} CACHE STRING "ADIOS compilation flags")
	mark_as_advanced(ADIOS_COMP_FLAGS)
	if (MPI_FOUND)
		set(ADIOS_LINK_FLAGS $ENV{ADIOS_CLIB} CACHE INTERNAL "ADIOS link flags including link dir and libs")
		if ( NOT ADIOS_LINK_FLAGS )
			EXECUTE_PROCESS(COMMAND ${ADIOS_DIR}/bin/adios_config -l
				OUTPUT_VARIABLE ADIOS_LINK_FLAGS
				OUTPUT_STRIP_TRAILING_WHITESPACE)
#			message(STATUS "ADIOS LIB FLAGS = ${ADIOS_LINK_FLAGS}")
		endif(NOT ADIOS_LINK_FLAGS)
	else(MPI_FOUND)
		set(ADIOS_LINK_FLAGS $ENV{ADIOS_SEQ_CLIB})
		if ( NOT ADIOS_LINK_FLAGS )
			EXECUTE_PROCESS(COMMAND ${ADIOS_DIR}/bin/adios_config -l -s
				OUTPUT_VARIABLE ADIOS_LINK_FLAGS
				OUTPUT_STRIP_TRAILING_WHITESPACE)
#			message(STATUS "ADIOS LIB FLAGS = ${ADIOS_LINK_FLAGS}")
		endif(NOT ADIOS_LINK_FLAGS)
	endif(MPI_FOUND)

    # parse out the libs and libdirectories.
    string(REGEX MATCHALL "-L[^ ]+" TEMP ${ADIOS_LINK_FLAGS})
    string(REPLACE "-L" " " TEMP ${TEMP})
    string(STRIP ${TEMP} TEMP)
	string(REPLACE " " ";" TEMP ${TEMP})
	set(ADIOS_LIB_DIRS ${TEMP} CACHE STRING "paths where to search for adios libraries and referenced libraries")  
	mark_as_advanced(ADIOS_LIB_DIRS)
#    message(STATUS " lib dirs = ${ADIOS_LIB_DIRS}")
    string(REGEX MATCHALL "-l[^ ]+" TEMP ${ADIOS_LINK_FLAGS})
    string(REPLACE "-l" " " TEMP ${TEMP})
    string(STRIP ${TEMP} TEMP)
	string(REPLACE " " ";" TEMP ${TEMP})
    set(ADIOS_LIBS ${TEMP} CACHE STRING "ADIOS libraries and referenced libraries")  
    mark_as_advanced(ADIOS_LIBS)
#    message(STATUS "libraries = ${ADIOS_LIBS}")
    
    link_directories(${ADIOS_LIB_DIRS})

    set(ADIOS_FOUND 1)
    else(ADIOS_DIR)
	message(SEND_ERROR "ERROR: Cannot find ADIOS root.  please specify manually")
	set(ADIOS_FOUND 0)
	endif(ADIOS_DIR)
else(USE_ADIOS)
    set(ADIOS_FOUND 0)

endif(USE_ADIOS)


# Make sure the compiler can find the includes
include_directories(${Segmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Segmentation_SOURCE_DIR})


if(ADIOS_FOUND)
	### copy over the adios xml files
	### and generate c source from ADIOS_XML files
	FILE(GLOB ADIOS_XMLS_IN RELATIVE ${PROJECT_SOURCE_DIR} "*.xml.in" )
	set(ADIOS_XML_DIR ${EXECUTABLE_OUTPUT_PATH}/../adios_xml)
	#message("adios xml dir = ${ADIOS_XML_DIR}")
	make_directory(${ADIOS_XML_DIR})

	# at this point, we need to have LD_LIBRARY_PATH set in order for the adios_lint (and gpp.py) to work.
	# this appears not to work.  so we would need to set this in the LD_LIBRARY_PATH in the bashrc script.
#	string(REPLACE ";" ":" ADIOS_LD_DIRS ${ADIOS_LIB_DIRS})
#	set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:${ADIOS_LD_DIRS}")

	# create a list of suffixes.
	set(ADIOS_TRANSPORTS NULL POSIX MPI_LUSTRE MPI_AMR MPI)
	set(NULL_TRANSPORT_OPTION "")
	set(POSIX_TRANSPORT_OPTION "")
	set(MPI_TRANSPORT_OPTION "")
	set(MPI_LUSTRE_TRANSPORT_OPTION "block_size=68157440")
	set(MPI_AMR_TRANSPORT_OPTION "num_aggregator=16;osts_to_skip=0-100;num_ost=200")
	
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

	# collect all the generated .ch files.	
	FILE(GLOB ADIOS_CHS RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "*.ch" )

	# add several header fields that are normally not installed with adios
	include_directories(${PROJECT_SOURCE_DIR}/adios-include)

endif(ADIOS_FOUND)



#### Segmentation code
#### ADIOS code
set(io_libsrcs SCIOUtilsLogger.cpp)
set(io_gpulibsrcs SCIOUtilsCVImageIO.cpp)
set(io_libname nu-io)
# Add rule to generate execution engine, as a shared library (.so)
set(TARGET_DEFS "")
set(TARGET_COMP_FLAGS "-fPIC")
if(CUDA_FOUND)
	cuda_add_library("${io_libname}" SHARED ${io_gpulibsrcs} ${io_libsrcs})
	set(TARGET_DEFS "${TARGET_DEFS} WITH_CUDA")
else(CUDA_FOUND)
	add_library("${io_libname}" SHARED ${io_gpulibsrcs} ${io_libsrcs})
endif(CUDA_FOUND)

if (MPI_FOUND)	
	set(TARGET_DEFS "${TARGET_DEFS} WITH_MPI")
	set(TARGET_COMP_FLAGS "${TARGET_COMP_FLAGS} ${MPI_COMPILE_FLAGS}")
	set_target_properties(${io_libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")			
endif(MPI_FOUND)
set_target_properties(${io_libname} PROPERTIES COMPILE_FLAGS "${TARGET_COMP_FLAGS}")
set_target_properties(${io_libname} PROPERTIES COMPILE_DEFINITIONS "${TARGET_DEFS}")
target_link_libraries(${io_libname} segment)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
	


#### ADIOS code
if (ADIOS_FOUND)
	set(adios_gpulibsrcs SCIOUtilsADIOS.cpp)
	set(adios_libname scio-adios)
	
	# Add rule to generate execution engine, as a shared library (.so)
	set(TARGET_DEFS "WITH_ADIOS")
	set(TARGET_COMP_FLAGS "${ADIOS_COMP_FLAGS} -fPIC")

	if(CUDA_FOUND)
		cuda_add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
		set(TARGET_DEFS "${TARGET_DEFS};WITH_CUDA")
	else(CUDA_FOUND)
		add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
	endif(CUDA_FOUND)

	if (MPI_FOUND)	
		set(TARGET_DEFS "${TARGET_DEFS};WITH_MPI")
		set(TARGET_COMP_FLAGS "${TARGET_COMP_FLAGS} ${MPI_COMPILE_FLAGS}")
		set_target_properties(${adios_libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")			
	endif(MPI_FOUND)
#	message(STATUS "SCIO_ADIOS DEFS = ${TARGET_DEFS}")
#	message(STATUS "SCIO_ADIOS COMPILE_FLAGS = ${TARGET_COMP_FLAGS}")
	
	set_target_properties(${adios_libname} PROPERTIES COMPILE_DEFINITIONS "${TARGET_DEFS}") 
	set_target_properties(${adios_libname} PROPERTIES COMPILE_FLAGS "${TARGET_COMP_FLAGS}") 
	
	target_link_libraries(${adios_libname} segment ${io_libname} ${ADIOS_LIBS})
	include_directories(${CMAKE_CURRENT_BINARY_DIR})
	
endif(ADIOS_FOUND)


# Source files used by Segmentation
set(libsrcs SCIOHistologicalEntities.cpp )
set(gpulibsrcs SCIOHistologicalEntitiesGPU.cpp)
set(libname scio-segment)
# Add rule to generate execution engine, as a shared library (.so)
if(CUDA_FOUND)
	cuda_add_library(${libname} SHARED ${libsrcs} ${gpulibsrcs})
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_CUDA") 
else(CUDA_FOUND)
	add_library(${libname} SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)
# Link opencv stuff to the execution engine
target_link_libraries(${libname} segment ${io_libname})






set(programs nu-segment-scio)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} ${libname} segment ${io_libname})

	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI") 
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} ${MPI_LIBRARIES})
	endif(MPI_FOUND)
endforeach(program)


if (ADIOS_FOUND)

	# Link the executable to the Features Computation library
	set(TARGET_DEFS "WITH_ADIOS")
	set(TARGET_COMP_FLAGS ${ADIOS_COMP_FLAGS})
	if (MPI_FOUND)	
		set(TARGET_DEFS "${TARGET_DEFS};WITH_MPI")
		set(TARGET_COMP_FLAGS "${TARGET_COMP_FLAGS} ${MPI_COMPILE_FLAGS}")
	endif(MPI_FOUND)
#	message(STATUS "PROGRAM DEFS = ${TARGET_DEFS}")
#	message(STATUS "PROGRAM COMPILE_FLAGS = ${TARGET_COMP_FLAGS}")


	set(programs nu-segment-scio-adios)
	foreach(program ${programs})
		# Add an executable to be built from the files listed
		add_executable(${program} ${program}.cpp)

		include_directories(${Segmentation_SOURCE_DIR})
		link_directories(${Segmentation_BINARY_DIR})
	
		# Link the executable to the Features Computation library
		target_link_libraries(${program} ${libname} segment ${io_libname} ${adios_libname})

		if (MPI_FOUND)	
			target_link_libraries(${program} ${MPI_LIBRARIES})
			set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
		endif(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "${TARGET_DEFS}")
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${TARGET_COMP_FLAGS}")
		
	endforeach(program)

	set(programs nu-features-adios)
	foreach(program ${programs})
		# Add an executable to be built from the files listed
		add_executable(${program} ${program}.cpp)
	
		include_directories(${Segmentation_SOURCE_DIR})
		link_directories(${Segmentation_BINARY_DIR})
		include_directories(${FeaturesComputation_SOURCE_DIR})
		link_directories(${FeaturesComputation_BINARY_DIR})
		
		# Link the executable to the Features Computation library
		target_link_libraries(${program} features segment ${io_libname} ${libname} ${adios_libname})
	
		if (MPI_FOUND)	
			target_link_libraries(${program} ${MPI_LIBRARIES})
			set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
		endif(MPI_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_DEFINITIONS "${TARGET_DEFS}")
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${TARGET_COMP_FLAGS}")
	endforeach(program)

endif(ADIOS_FOUND)



add_subdirectory(test)
