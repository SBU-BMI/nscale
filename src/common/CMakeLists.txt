project( cci_common )

# The version number
set(cci_common_VERSION_MAJOR 1)
set(cci_common_VERSION_MINOR 0)

# Make sure the compiler can find the includes
#include_directories(${Common_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
#link_directories(${Runtime_BINARY_DIR})


# HDF5
#### for Keeneland, and possibly Jaguar
#### the default CMake module SHOULD work, if a hint of HDF5_ROOT is provided in the environment
#### this is where bin/h5cc is.  Note that on Keeneland, there is a /usr/bin/h5cc which is not right.
####  also note that HDF5 is not compiled with CMake on keeneland, so there is not HDF5-config.cmake file for the 
####  FindHDF5.cmake file in the HDF5 source directory to work against.
set(ENV{HDF5_ROOT} $ENV{HDF5_DIR})
set(HDF5_USE_STATIC_LIBRARIES OFF)
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
    set(libsrcs Debug.cpp FileUtils.cpp Logger.cpp h5utils.cpp)
        
else(HDF5_FOUND)
    set(libsrcs Debug.cpp FileUtils.cpp Logger.cpp)
        
endif(HDF5_FOUND)



# Source files used by NScale_Common
set(libname cci_common)

# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs} )

if(MPI_FOUND)
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
	set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
	target_link_libraries(${libname} ${MPI_LIBRARIES})
elseif(MPI_FOUND)
    message(WARNING "MPI was not found.  libcci_common may be incomplete" )
endif(MPI_FOUND)

if (HDF5_FOUND)
    target_link_libraries(${libname} ${HDF5_LIBRARIES})
endif(HDF5_FOUND)


if(BUILD_TEST_APPLICATIONS)
    add_subdirectory(test/src)
endif(BUILD_TEST_APPLICATIONS)
