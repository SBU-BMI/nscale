project( cci_common )

# The version number
set(cci_common_VERSION_MAJOR 1)
set(cci_common_VERSION_MINOR 0)

# Make sure the compiler can find the includes
#include_directories(${Common_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
#link_directories(${Runtime_BINARY_DIR})

if (HDF5_FOUND)
    set(libsrcs Debug.cpp FileUtils.cpp Logger.cpp h5utils.cpp MathUtils.cpp)
else(HDF5_FOUND)
    set(libsrcs Debug.cpp FileUtils.cpp Logger.cpp MathUtils.cpp)    
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
