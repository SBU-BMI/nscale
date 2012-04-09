project( SCIOSegmentation )

# The version number
set(SCIOSegmentation_VERSION_MAJOR 1)
set(SCIOSegmentation_VERSION_MINOR 0)

#### ADIOS
if (MPI_FOUND)
	set(WITH_ADIOS OFF CACHE BOOL "Build with ADIOS support")
	if (WITH_ADIOS)
	    find_path(ADIOS_INCLUDE_DIR adios.h
		HINTS /usr/include /usr/local/include /usr/local/share/include /opt/local/include /usr/local/share/adios/include
		DOC "Include path to adios.h file")
	    find_library(ADIOS_LIBRARY adios libadios
		HINTS ${ADIOS_INCLUDE_DIR}/../lib
		DOC "Adios Library file")
	    GET_FILENAME_COMPONENT(ADIOS_LIBRARY_DIR ${ADIOS_LIBRARY} PATH)
		include_directories(${ADIOS_INCLUDE_DIR})
		link_directories(${ADIOS_LIBRARY_DIR})
	    set(ADIOS_LIBRARIES "m;mxml;pthread;adios")
	    set(ADIOS_FOUND 1)
	else(WITH_ADIOS)
	    set(ADIOS_FOUND 0)

	endif(WITH_ADIOS)
endif(MPI_FOUND)


# Make sure the compiler can find the includes
include_directories(${Segmentation_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${Segmentation_SOURCE_DIR})


#### Segmentation code

# Source files used by Segmentation
set(libsrcs SCIOUtilsLogger.cpp SCIOHistologicalEntities.cpp )
set(gpulibsrcs SCIOUtilsCVImageIO.cpp SCIOHistologicalEntitiesGPU.cpp)

set(libname scio-segment)
# Add rule to generate execution engine, as a shared library (.so)
if(CUDA_FOUND)
	cuda_add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS -DHAVE_CUDA) 
else(CUDA_FOUND)
	add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)
# Link opencv stuff to the execution engine
target_link_libraries("${libname}" segment ${OpenCV_LIBS})



#### ADIOS code
set(adios_gpulibsrcs SCIOUtilsADIOS.cpp)
set(adios_libname scio-adios)
if (ADIOS_FOUND)
# Add rule to generate execution engine, as a shared library (.so)
	if(CUDA_FOUND)
		cuda_add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
		set_target_properties(${adios_libname} PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_ADIOS -fPIC") 
	else(CUDA_FOUND)
		add_library("${adios_libname}" SHARED ${adios_gpulibsrcs})
		set_target_properties(${adios_libname} PROPERTIES COMPILE_FLAGS "-DHAVE_ADIOS -fPIC")
	endif(CUDA_FOUND)
	target_link_libraries(${adios_libname} segment ${ADIOS_LIBRARIES} ${OpenCV_LIBS})
endif(ADIOS_FOUND)




set(programs nu-segment-scio nu-segment-scio-adios test-adios test-mpi test-threads)
foreach(program ${programs})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	target_link_libraries(${program} scio-segment segment ${OpenCV_LIBS})

	if (ADIOS_FOUND)
		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI -DWITH_ADIOS")
  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
		target_link_libraries(${program} scio-adios ${ADIOS_LIBRARIES} ${MPI_LIBRARIES})
	else(ADIOS_FOUND)
		if(MPI_FOUND)
	  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
	  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
			target_link_libraries(${program} ${MPI_LIBRARIES})
		endif(MPI_FOUND)
	endif(ADIOS_FOUND)
endforeach(program)


### copy over the adios xml files
FILE(GLOB ADIOS_XMLS RELATIVE ${PROJECT_SOURCE_DIR} "*.xml" )
set(ADIOS_XML_DIR ${CMAKE_BINARY_DIR}/adios_xml)
#message("adios xml dir = ${ADIOS_XML_DIR}")
make_directory(${ADIOS_XML_DIR})
foreach(ADIOS_XML ${ADIOS_XMLS})
	#message("copy from ${PROJECT_SOURCE_DIR}/${ADIOS_XML} to ${ADIOS_XML_DIR}/${ADIOS_XML}")
	configure_file(	${PROJECT_SOURCE_DIR}/${ADIOS_XML} ${ADIOS_XML_DIR}/${ADIOS_XML} COPY_ONLY)
endforeach(ADIOS_XML)
