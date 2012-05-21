add_subdirectory(execEngine)
add_subdirectory(features-cpu)
add_subdirectory(features)
add_subdirectory(segment)
### set to compile the pipeline directory
set(PIPELINE OFF CACHE BOOL "Building Pipeline, which requires HDF5 1.8.7")
if (PIPELINE)
	add_subdirectory(pipeline)
endif(PIPELINE)

#### set to compile the SC-IO directory
set(SC_IO OFF CACHE BOOL "Building experimental code for SC12 IO studies")
if (SC_IO)
	add_subdirectory(studies/SC-IO)
endif(SC_IO)

#add_subdirectory(tasks)
