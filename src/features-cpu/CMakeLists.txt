project( FeaturesComputation )

# The version number
set(FeaturesComputation_VERSION_MAJOR 1)
set(FeaturesComputation_VERSION_MINOR 1)

# Source files used by Execution Engine
set(libsrcs Contour.cpp Blob.cpp RegionalMorphologyAnalysis.cpp DrawAuxiliar.cpp Operators.cpp)

set(libname features)

# Add rule to generate execution engine, as a shared library (.so)
add_library("${libname}" SHARED ${libsrcs})

# Link Pthreads stuff to the execution engine
target_link_libraries("${libname}" ${OpenCV_LIBS})

# if build same applications is set, than build them: currently just one
if(BUILD_EXAMPLE_APPLICATIONS)
	add_subdirectory("examples/")
endif(BUILD_EXAMPLE_APPLICATIONS)
