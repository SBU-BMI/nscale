cmake_minimum_required(VERSION 2.8)

PROJECT( NSCALE )
# The version number
set(NSCALE_VERSION_MAJOR 1 CACHE INTERNAL "major version")
set(NSCALE_VERSION_MINOR 0 CACHE INTERNAL "minor version")

#message(STATUS "CMAKE binary dir location: ${CMAKE_BINARY_DIR}")
#message(STATUS "PROJECT binary dir location: ${PROJECT_BINARY_DIR}")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
    set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()


#to avoid confusion with macro in windows.h
IF( MSVC )
add_definitions( /DNOMINMAX ) 
ENDIF()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# Default: dynamic libraries
# ----------------------------------------------------------------------------
#set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")

# ----------------------------------------------------------------------------
# Include profiling?
# Default: NO
# ----------------------------------------------------------------------------
set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")

# Profiling?
if(ENABLE_PROFILING)
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
else()
    # Remove unreferenced functions: function level linking
    if(NOT APPLE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
    endif()
endif()

# ----------------------------------------------------------------------------
#                       common libaries needed by all
# ----------------------------------------------------------------------------

#### MPI
set(USE_MPI OFF CACHE BOOL "Build with MPI support")
if (USE_MPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
else(USE_MPI)
	set(MPI_FOUND 0)
endif(USE_MPI)

#### OpenMP
set(USE_OPENMP ON CACHE BOOL "Build with OpenMP support")
if (USE_OPENMP)
    include(FindOpenMP)

	 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(USE_OPENMP)


#### for compression
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
	include_directories(${ZLIB_INCLUDE_DIRS})
endif(ZLIB_FOUND)

# HDF5
set(USE_HDF5 OFF CACHE BOOL "Include HDF5 support")
#### for Keeneland, and possibly Jaguar
#### the default CMake module SHOULD work, if a hint of HDF5_ROOT is provided in the environment
#### this is where bin/h5cc is.  Note that on Keeneland, there is a /usr/bin/h5cc which is not right.
####  also note that HDF5 is not compiled with CMake on keeneland, so there is not HDF5-config.cmake file for the 
####  FindHDF5.cmake file in the HDF5 source directory to work against.
if(USE_HDF5)
    set(ENV{HDF5_ROOT} $ENV{HDF5_DIR})
    set(HDF5_USE_STATIC_LIBRARIES OFF)
    find_package(HDF5 REQUIRED)
else(USE_HDF5)
    set(HDF5_FOUND 0)    
endif(USE_HDF5)

if (HDF5_FOUND)
	message(STATUS "HDF5_DEF = ${HDF5_DEFINITIONS}")
	message(STATUS "HDF5LIBRARY_DIRS flags: ${HDF5_LIBRARY_DIRS}")
	message(STATUS "HDF5INCLUDE_DIRS include: ${HDF5_INCLUDE_DIRS}")
	message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}")
	message(STATUS "HDF5 HL libs: ${HDF5_HL_LIBRARIES}")
	message(STATUS "HDF5C libs: ${HDF5_C_LIBRARIES}")
	message(STATUS "HDF5CXX libs: ${HDF5_CXX_LIBRARIES}")        
	
	include_directories(${HDF5_INCLUDE_DIRS})
	add_definitions(${HDF5_DEFINITIONS})
	link_directories(${HDF5_LIBRARY_DIRS})

# HDF5_LIBRARIES as defined on Keeneland is broken.  it does not respect the HDF5_DIR.
# besides there should not be absolute paths here anyways.
	set(HDF5_LIBRARIES "hdf5_hl;hdf5;z;m")
	message(STATUS "Redefined HDF5 libs: ${HDF5_LIBRARIES}")
    
else(HDF5_FOUND)
    message(STATUS "HDF5 not found")
endif(HDF5_FOUND)

# ----------------------------------------------------------------------------
#                       PROCESS SUBDIRECTORIES:
# ----------------------------------------------------------------------------
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

set(CMAKE_EXECUTABLE_SUFFIX ".exe")

# Check if the user want to build test applications
option (BUILD_TEST_APPLICATIONS "Inform whether test applications should be built" ON)
# Check if the user want to build sample applications
option (BUILD_EXAMPLE_APPLICATIONS "Inform whether sample applications should be built" ON)

add_subdirectory(src)
add_subdirectory(scripts)
