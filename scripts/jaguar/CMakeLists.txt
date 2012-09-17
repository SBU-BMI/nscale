project( Jaguar_qsub_scripts )

set(Jaguar_qsub_scripts_VERSION_MAJOR 1)

set(SCRIPT_OUT_DIR ${EXECUTABLE_OUTPUT_PATH}/../scripts)
FILE(MAKE_DIRECTORY ${SCRIPT_OUT_DIR})

set(PROC_COUNTS 10240 5120 2560)
set(WALLTIME_STRONG_10240 01:00:00)
set(WALLTIME_STRONG_5120 02:00:00)
set(WALLTIME_STRONG_2560 03:00:00)
set(WALLTIME_WEAK 01:00:00)

set(TILE_COUNT_PER_PROC 7)
set(TILE_COUNT_ALL 100000)

set(ADIOS_TRANSPORTS na-NULL na-POSIX NULL POSIX MPI MPI_LUSTRE MPI_AMR)
set(na-NULL_IO_GROUP 1)
set(na-POSIX_IO_GROUP 1)
set(NULL_IO_GROUP 1)
set(POSIX_IO_GROUP 1 16 -1)
set(MPI_IO_GROUP 16 -1)
set(MPI_LUSTRE_IO_GROUP 16 -1)
set(MPI_AMR_IO_GROUP -1)

set(IO_INTERLEAVE 1)
set(IO_COUNT_OST 200)
set(IO_COUNT_ALL 300)
#set(IO_COUNT_FRACTIONS 3/4 1/2 1/4)

set(BUFFER_SIZE 4)
set(PLATFORM jaguar)
set(NEGONE -1)
set(IO_SIZE 0)

foreach(PROC_COUNT ${PROC_COUNTS})

	##### STRONG scaling version
	set(TILE_COUNT ${TILE_COUNT_ALL})
    set(WALL_TIME ${WALLTIME_STRONG_${PROC_COUNT}})

    # generate the baseline set of scripts
    set(RUN_NAME baseline.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT})
    configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

    foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
    	foreach (ADIOS_TRANSPORT_GROUPSIZE ${${ADIOS_TRANSPORT}_IO_GROUP})
    	# -1 accepted for co-loc
    
    	    # generate the adios co-loc set of scripts
            set(RUN_NAME coloc.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
            # generate the adios separate IO node set of scripts.
        	set(IO_SIZE ${IO_COUNT_ALL})
        	if ("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
        		set(ADIOS_TRANSPORT_GROUPSIZE ${IO_SIZE})
        	endif("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
                
            set(RUN_NAME separate.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
    	endforeach(ADIOS_TRANSPORT_GROUPSIZE)
    endforeach(ADIOS_TRANSPORT)



####################
# WEAK scaling version

	##### STRONG scaling version
	math(EXPR TILE_COUNT "${TILE_COUNT_PER_PROC}*${PROC_COUNT}")
    set(WALL_TIME ${WALLTIME_WEAK})

    # generate the baseline set of scripts
    set(RUN_NAME baseline.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT})
    configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

    foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
    	foreach (ADIOS_TRANSPORT_GROUPSIZE ${${ADIOS_TRANSPORT}_IO_GROUP})
    	# -1 accepted for co-loc
    
    	    # generate the adios co-loc set of scripts
            set(RUN_NAME coloc.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
            # generate the adios separate IO node set of scripts.
        	set(IO_SIZE ${IO_COUNT_ALL})
        	if ("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
        		set(ADIOS_TRANSPORT_GROUPSIZE ${IO_SIZE})
        	endif("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
                 
            set(RUN_NAME separate.${PLATFORM}.p${PROC_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)
    
    	endforeach(ADIOS_TRANSPORT_GROUPSIZE)
    endforeach(ADIOS_TRANSPORT)



endforeach(PROC_COUNT)

###########
# submission scripts

# baseline_weak
# adios_weak
# 

set(SUB_NODE_COUNTS ${PROC_COUNTS})
configure_file(${PROJECT_SOURCE_DIR}/baseline.weak.submit.sh.in ${SCRIPT_OUT_DIR}/jaguar.baseline.weak.submit.sh @ONLY)
set(SUB_ADIOS_TRANSPORTS POSIX MPI_LUSTRE MPI_AMR)
configure_file(${PROJECT_SOURCE_DIR}/adios.coloc.weak.submit.sh.in ${SCRIPT_OUT_DIR}/jaguar.adios.coloc.weak.submit.sh @ONLY)
set(SUB_ADIOS_TRANSPORTS na-POSIX POSIX MPI_LUSTRE MPI_AMR)
configure_file(${PROJECT_SOURCE_DIR}/adios.separate.weak.submit.sh.in ${SCRIPT_OUT_DIR}/jaguar.adios.separate.weak.submit.sh @ONLY)
