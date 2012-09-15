project( scripts )

set(CONFIG_SCRIPTS OFF CACHE BOOL "Configure scripts for PBS submission")

if(CONFIG_SCRIPTS) 

set(JOB_STATUS_EMAIL CACHE STRING "email address used by PBS to notify about job status")
set(SCRIPT_TEMPLATE_DIR CACHE STRING "directory where the job submission script templates are. knows about jaguar, keeneland, and mpi")
message(STATUS "get script dir ${SCRIPT_TEMPLATE_DIR}")

if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${SCRIPT_TEMPLATE_DIR})
	message(STATUS "adding subdir ${SCRIPT_TEMPLATE_DIR}")
    add_subdirectory(${SCRIPT_TEMPLATE_DIR})
endif(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${SCRIPT_TEMPLATE_DIR})

endif(CONFIG_SCRIPTS)


