add_subdirectory(common)

set(NS_MPI_Runtime OFF CACHE BOOL "Building MPI runtime")
if (NS_MPI_Runtime)

    #### BZIP Library (for boost?)
    find_package(BZip2)
    if (BZIP2_FOUND)
        include_directories( ${BZIP2_INCLUDE_DIR} )
        set (LIBS ${LIBS} ${BZIP2_LIBRARIES})
    endif (BZIP2_FOUND)
    
    #### Boost Library
    set(Boost_USE_MULTITHREADED OFF)
    set(Boost_ADDITIONAL_VERSIONS "1.52" "1.52.0")
    set(Boost_DETAILED_FAILURE_MSG ON)
    find_path(BOOST_ROOT NAMES include/boost/circular_buffer.hpp
    	HINTS /usr /usr/local /usr/local/share /opt/local ENV BOOST_HOME
    	DOC "path to Boost root directory")
    find_package( Boost REQUIRED COMPONENTS program_options )
    # must specify the components otherwise it's a header only usage
    if (Boost_FOUND)
        include_directories( ${Boost_INCLUDE_DIRS} )
        link_directories( ${Boost_LIBRARY_DIRS} )
        message(STATUS "Boost libs: " ${Boost_LIBRARIES})
    endif(Boost_FOUND)


	add_subdirectory(runtime)
endif(NS_MPI_Runtime)

set(NS_EXECENGINE OFF CACHE BOOL "Building ExecEngine")
if (NS_EXECENGINE)
    # Find threads package which is linked to the execution engine library
    find_package( Threads REQUIRED )

    add_subdirectory(execEngine)
endif(NS_EXECENGINE)

set(NS_FEATURES OFF CACHE BOOL "Building Features and FeaturesCPU")
set(NS_SEGMENT OFF CACHE BOOL "Building Segmentation")
set(NS_PIPELINE OFF CACHE BOOL "Building Pipeline, which requires HDF5 1.8.7")
set(NS_AdvIO OFF CACHE BOOL "Building experimental code for Advanced IO studies")
#set(NS_Tasks OFF CACHE BOOL "Building experimental code for Task Centric Pipeline")

if(NS_FEATURES OR NS_SEGMENT)
    #### OpenCV
    find_path(OpenCV_CMAKE_DIR NAMES FindOpenCV.cmake OpenCVConfig.cmake
    	HINTS ${CMAKE_MODULE_PATH} ENV OPENCV_HOME
    	DOC "path to FindOpenCV.cmake or OpenCVConfig.cmake file") 
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${OpenCV_CMAKE_DIR})
    find_package( OpenCV REQUIRED )
    #message(STATUS "openCV lib = ${OpenCV_LIBS}")
    
    #### CUDA
    set(USE_CUDA OFF CACHE BOOL "Include NVidia Cuda Runtime support")
    if(USE_CUDA)
        find_package(CUDA 4.0)
        
        if(CUDA_FOUND) 		
            message(STATUS "CUDA detected: " ${CUDA_VERSION})
    
            # These vars will be processed in other scripts
         	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math -arch=sm_20 -DPREFIX_SUM)
         #	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -G)  # too slow...
           set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -DTHRUST_DEBUG)
            message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")
    	add_definitions( -DWITH_CUDA )
        endif(CUDA_FOUND)
    else(USE_CUDA)
    	set(CUDA_FOUND 0)
    endif(USE_CUDA)
    
endif(NS_FEATURES OR NS_SEGMENT)

if (NS_FEATURES)
    add_subdirectory(features-cpu)
    add_subdirectory(features)
endif(NS_FEATURES)
    

if (NS_SEGMENT)
    add_subdirectory(segment)
endif(NS_SEGMENT)

### set to compile the pipeline directory

if (NS_PIPELINE)
    if(NS_SEGMENT AND NS_FEATURES AND USE_HDF5) 
    	add_subdirectory(pipeline)
    else(NS_SEGMENT AND NS_FEATURES AND USE_HDF5)
        message(SEND_ERROR "PIPELINE requires SEGMENT, FEATURES, and USE_HDF5")
    endif(NS_SEGMENT AND NS_FEATURES AND USE_HDF5)
endif(NS_PIPELINE)

#### set to compile the SC-IO directory

if (NS_AdvIO)
    if(NS_SEGMENT) 
    	add_subdirectory(studies/SC-IO)
    else(NS_SEGMENT)
        message(SEND_ERROR "AdvIO requires SEGMENT")
    endif(NS_SEGMENT)
endif(NS_AdvIO)

#if (NS_Tasks)
#    if(SEGMENT AND FEATURES AND HDF5_FOUND) 
#    	add_subdirectory(tasks)
#    else(SEGMENT AND FEATURES AND HDF5_FOUND)
#        message(SEND_ERROR "Tasks requires SEGMENT and FEATURES")
#    endif(SEGMENT AND FEATURES AND HDF5_FOUND)
#endif(NS_Tasks)
#

if(BUILD_EXAMPLE_APPLICATIONS)
    if(NS_AdvIO AND NS_MPI_Runtime AND NS_SEGMENT) 
    	add_subdirectory(runtime-examples)
    endif(NS_AdvIO AND NS_MPI_Runtime AND NS_SEGMENT)
endif(BUILD_EXAMPLE_APPLICATIONS)
