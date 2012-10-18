project( NScale_Common )

# The version number
set(NScale_Common_VERSION_MAJOR 1)
set(NScale_Common_VERSION_MINOR 0)

# Make sure the compiler can find the includes
#include_directories(${Runtime_SOURCE_DIR})

# Make sure the compiler can find the library in Execution Engine directory
#link_directories(${Runtime_BINARY_DIR})

# Source files used by NScale_Common
#set(libsrcs HistologicalEntities.cpp MorphologicOperations.cpp PixelOperations.cpp NeighborOperations.cpp FileUtils.cpp ConnComponents.cpp )

#set(libname nscale_common)

# Add rule to generate execution engine, as a shared library (.so)
#add_library("${libname}" SHARED ${libsrcs} )

#if(MPI_FOUND)
#	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_MPI")
#	set_target_properties(${libname} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}") 
#	set_target_properties(${libname} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
#	target_link_libraries(${libname} ${MPI_LIBRARIES})
#endif(MPI_FOUND)
