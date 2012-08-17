project( mpi_scripts )

set(mpi_scripts_VERSION_MAJOR 1)
set(mpi_scripts_VERSION_MINOR 0)

set(SCRIPT_OUT_DIR ${EXECUTABLE_OUTPUT_PATH}/../scripts)
FILE(MAKE_DIRECTORY ${SCRIPT_OUT_DIR})

set(PROC_COUNTS 128 64 32)
set(TILE_COUNTS 10000)

set(ADIOS_TRANSPORTS MPI_AMR)
set(IO_INTERLEAVE 4)
set(IO_COUNT_FRACTIONS 3/4 1/2 1/4)

set(BUFFER_SIZE 4)
set(PLATFORM cci)
set(IO_SIZE 0)

foreach(PROC_COUNT ${PROC_COUNTS})
    foreach(TILE_COUNT ${TILE_COUNTS})
        # generate the baseline set of scripts
        set(RUN_NAME baseline.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT})
        configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
        # generate the adios co-loc set of scripts
        foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
            set(RUN_NAME co-loc.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

            # generate the adios separate IO node set of scripts.
            foreach(IO_COUNT_FRACTION ${IO_COUNT_FRACTIONS})
                math(EXPR IO_SIZE "${PROC_COUNT}*${IO_COUNT_FRACTION}")
                
                set(RUN_NAME separate.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE})
                configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

            endforeach(IO_COUNT_FRACTION)

        endforeach(ADIOS_TRANSPORT)


    endforeach(TILE_COUNT)
endforeach(PROC_COUNT)

configure_file(${PROJECT_SOURCE_DIR}/mpi_submit.sh.in ${SCRIPT_OUT_DIR}/mpi_submit.sh @ONLY)
