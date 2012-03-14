add_subdirectory(execEngine)
add_subdirectory(features-cpu)
add_subdirectory(segment)
add_subdirectory(pipeline)
#add_subdirectory(tasks)

#### set to compile the SC-IO directory
set(SC_IO OFF CACHE BOOL "Building experimental code for SC12 IO studies")
if (SC_IO)
	add_subdirectory(studies/SC-IO)
endif(SC_IO)

