project( Jaguar_qsub_scripts )

set(Jaguar_qsub_scripts_VERSION_MAJOR 1)
set(Jaguar_qsub_scripts_VERSION_MINOR 0)

set(SCRIPT_OUT_DIR ${EXECUTABLE_OUTPUT_PATH}/../scripts)
FILE(MAKE_DIRECTORY ${SCRIPT_OUT_DIR})

set(PROC_COUNTS 10240 5120 2560)
set(WALLTIME_10240 02:00:00)
set(WALLTIME_5120 03:00:00)
set(WALLTIME_2560 05:00:00)

set(TILE_COUNTS 100000)

set(ADIOS_TRANSPORTS na-NULL na-POSIX NULL POSIX MPI MPI_LUSTRE MPI_AMR)
set(IO_INTERLEAVE 16)
set(IO_COUNT_FRACTIONS 3/4 1/2 1/4)

set(BUFFER_SIZE 4)
set(PLATFORM jaguar)
set(IO_SIZE 0)

set(JOB_STATUS_EMAIL CACHE STRING "email address used for PBS to notify about job status")

foreach(PROC_COUNT ${PROC_COUNTS})
    foreach(TILE_COUNT ${TILE_COUNTS})
        # generate the baseline set of scripts
        set(RUN_NAME baseline.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT})
	    set(WALL_TIME 01:00:00)
        configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
        # generate the adios co-loc set of scripts
        foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
            set(RUN_NAME co-loc.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE})
	        set(WALL_TIME 06:00:00)
            configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

            # generate the adios separate IO node set of scripts.
            foreach(IO_COUNT_FRACTION ${IO_COUNT_FRACTIONS})
                math(EXPR IO_SIZE "${PROC_COUNT}*${IO_COUNT_FRACTION}")
                
                set(RUN_NAME separate.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE})
		        set(WALL_TIME ${WALLTIME_${PROC_COUNT}})
                configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

            endforeach(IO_COUNT_FRACTION)

        endforeach(ADIOS_TRANSPORT)


    endforeach(TILE_COUNT)
endforeach(PROC_COUNT)

configure_file(${PROJECT_SOURCE_DIR}/warmup.sh ${SCRIPT_OUT_DIR}/warmup.sh COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/jaguar_submit.sh.in ${SCRIPT_OUT_DIR}/jaguar_submit.sh @ONLY)
