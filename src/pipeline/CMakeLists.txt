cmake_minimum_required(VERSION 2.6)
project( Pipeline )

# The version number
set(Pipeline_VERSION_MAJOR 1)
set(Pipeline_VERSION_MINOR 0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)



# Source files used by Execution Engine

# fix this properly later to correctly export and import other packages
include_directories(${Pipeline_SOURCE_DIR}/../segment)
include_directories(${Pipeline_SOURCE_DIR}/../features-cpu)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -L${Pipeline_SOURCE_DIR}/../segment")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -L${Pipeline_SOURCE_DIR}/../features-cpu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${Pipeline_SOURCE_DIR}/../segment")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${Pipeline_SOURCE_DIR}/../features-cpu")
# fix this properly later to correctly export and import other packages

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${Pipeline_SOURCE_DIR}/../../../textascale -L${Pipeline_SOURCE_DIR}/../../../textascale/nwkmeans/.libs")

#### OpenCV
find_package( OpenCV REQUIRED )

#### HDF5
find_package(HDF5 REQUIRED)
if (HDF5_FOUND)
	include_directories(${HDF5_INCLUDE_DIRS})
	add_definitions(${HDF5_DEFINITIONS})
message(STATUS "HDF5LIBRARY_DIRS flags: ${HDF5_LIBRARY_DIRS}")
message(STATUS "HDF5LIBRARY_DIRS include: ${HDF5_INCLUDE_DIRS}")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -L${HDF5_LIBRARY_DIRS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${HDF5_LIBRARY_DIRS}")
endif(HDF5_FOUND)

#### MPI
set(WITH_MPI OFF CACHE BOOL "Build with MPI support")
if (WITH_MPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
else(WITH_MPI)
	set(MPI_FOUND 0)
endif(WITH_MPI)

#### OpenMP
set(WITH_OPENMP OFF CACHE BOOL "Build with OpenMP support")
if (WITH_OPENMP)
    include(FindOpenMP)
    
	 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(WITH_OPENMP)


# Source files used by Execution Engine
set(libsrcs h5utils.cpp )

set(libname pipeline)
# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs} )
if(HDF5_FOUND)
	target_link_libraries(${libname} ${HDF5_LIBRARIES})
endif(HDF5_FOUND) 



set(CMAKE_EXECUTABLE_SUFFIX ".exe")


set(fprograms nu-features)
foreach(program ${fprograms})
	# Add an executable to be built from the files listed
	add_executable(${program} ${program}.cpp)

	# Link the executable to the Features Computation library
	
	target_link_libraries(${program} features)
	target_link_libraries(${program} segment)
	target_link_libraries(${program} pipeline)
	if(MPI_FOUND)
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
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
  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
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
#  		set_target_properties(${program} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DWITH_MPI") 
#  		set_target_properties(${program} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
#		target_link_libraries(${program} ${MPI_LIBRARIES})
#	endif(MPI_FOUND) 
#
#	if(HDF5_FOUND)
#		target_link_libraries(${program} ${HDF5_LIBRARIES})
#	endif(HDF5_FOUND) 
#	
#endforeach(program)


