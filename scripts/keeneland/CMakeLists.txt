project( keeneland_qsub_scripts )

set(keeneland_qsub_scripts_VERSION_MAJOR 1)
set(keeneland_qsub_scripts_VERSION_MINOR 0)

set(SCRIPT_OUT_DIR ${EXECUTABLE_OUTPUT_PATH}/../scripts)
FILE(MAKE_DIRECTORY ${SCRIPT_OUT_DIR})

set(NODE_COUNTS 8 16 32 60 100 120)
set(WALLTIME_STRONG_100 00:30:00)
set(WALLTIME_STRONG_60 01:00:00)
set(WALLTIME_STRONG_32 02:00:00)
set(WALLTIME_STRONG_16 03:00:00)
set(WALLTIME_STRONG_8 04:00:00)
set(WALLTIME_WEAK 00:20:00)
set(PROCS_PER_NODE 12)

# works out to be 25 tiles per proc.
set(TILE_COUNT_PER_PROC 25)
set(TILE_COUNT_ALL 36000)

set(ADIOS_TRANSPORTS na-NULL na-POSIX NULL POSIX MPI MPI_LUSTRE MPI_AMR)
set(na-NULL_IO_GROUP 1)
set(na-POSIX_IO_GROUP 1)
set(NULL_IO_GROUP 1)
set(POSIX_IO_GROUP 1 15 -1)
set(MPI_IO_GROUP 15 -1)
set(MPI_LUSTRE_IO_GROUP 15 -1)
set(MPI_AMR_IO_GROUP -1)

set(IO_INTERLEAVE 1)
set(IO_COUNT_OST 60)
set(IO_COUNT_ALL 60)
#set(IO_COUNT_FRACTIONS 3/4 1/2 1/4)

set(BUFFER_SIZE 4)
set(PLATFORM keeneland)
set(NEGONE -1)
set(IO_SIZE 0)

##############
# generate all the PBS scripts
foreach(NODE_COUNT ${NODE_COUNTS})
	math(EXPR PROC_COUNT "${NODE_COUNT}*12")

	##### STRONG scaling version
	set(TILE_COUNT ${TILE_COUNT_ALL})
    set(WALL_TIME ${WALLTIME_STRONG_${NODE_COUNT}})

    # generate the baseline set of scripts
    set(RUN_NAME baseline.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT})
    configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

    foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
	foreach (ADIOS_TRANSPORT_GROUPSIZE ${${ADIOS_TRANSPORT}_IO_GROUP})
	# -1 accepted for co-loc

	    # generate the adios co-loc set of scripts
        set(RUN_NAME coloc.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.is${ADIOS_TRANSPORT_GROUPSIZE})
        configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

        # generate the adios separate IO node set of scripts.
	set(IO_SIZE ${IO_COUNT_ALL})
	if ("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
		set(ADIOS_TRANSPORT_GROUPSIZE ${IO_SIZE})
	endif("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
            
            set(RUN_NAME separate.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

	endforeach(ADIOS_TRANSPORT_GROUPSIZE)
    endforeach(ADIOS_TRANSPORT)



####################
# WEAK scaling version

	##### STRONG scaling version
	math(EXPR TILE_COUNT "${TILE_COUNT_PER_PROC}*${PROC_COUNT}")
    set(WALL_TIME ${WALLTIME_WEAK})

    # generate the baseline set of scripts
    set(RUN_NAME baseline.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT})
    configure_file( ${PROJECT_SOURCE_DIR}/baseline.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

    foreach(ADIOS_TRANSPORT ${ADIOS_TRANSPORTS})
	foreach (ADIOS_TRANSPORT_GROUPSIZE ${${ADIOS_TRANSPORT}_IO_GROUP})
	# -1 accepted for co-loc

	    # generate the adios co-loc set of scripts
        set(RUN_NAME coloc.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.is${ADIOS_TRANSPORT_GROUPSIZE})
        configure_file( ${PROJECT_SOURCE_DIR}/adios-coloc.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

        # generate the adios separate IO node set of scripts.
	set(IO_SIZE ${IO_COUNT_ALL})
	if ("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
		set(ADIOS_TRANSPORT_GROUPSIZE ${IO_SIZE})
	endif("${ADIOS_TRANSPORT_GROUPSIZE}" STREQUAL "${NEGONE}")
             
            set(RUN_NAME separate.${PLATFORM}.n${NODE_COUNT}.f${TILE_COUNT}.${ADIOS_TRANSPORT}.b${BUFFER_SIZE}.io${IO_SIZE}-${IO_INTERLEAVE}.is${ADIOS_TRANSPORT_GROUPSIZE})
            configure_file( ${PROJECT_SOURCE_DIR}/adios-separate.sh.in ${SCRIPT_OUT_DIR}/${RUN_NAME}.sh @ONLY)

	endforeach(ADIOS_TRANSPORT_GROUPSIZE)
    endforeach(ADIOS_TRANSPORT)


endforeach(NODE_COUNT)

###########
# submission scripts

# baseline_weak
# adios_weak
# 

set(SUB_NODE_COUNTS 32 60)
configure_file(${PROJECT_SOURCE_DIR}/baseline.weak.submit.sh.in ${SCRIPT_OUT_DIR}/keeneland.baseline.weak.submit.sh @ONLY)
set(SUB_ADIOS_TRANSPORTS POSIX MPI_LUSTRE MPI_AMR)
configure_file(${PROJECT_SOURCE_DIR}/adios.coloc.weak.submit.sh.in ${SCRIPT_OUT_DIR}/keeneland.adios.coloc.weak.submit.sh @ONLY)
set(SUB_ADIOS_TRANSPORTS na-POSIX POSIX MPI_LUSTRE MPI_AMR)
configure_file(${PROJECT_SOURCE_DIR}/adios.separate.weak.submit.sh.in ${SCRIPT_OUT_DIR}/keeneland.adios.separate.weak.submit.sh @ONLY)
set(CAPABILITY_ADIOS_TRANSPORTS na-POSIX POSIX MPI_LUSTRE MPI_AMR)
configure_file(${PROJECT_SOURCE_DIR}/capability.submit.sh.in ${SCRIPT_OUT_DIR}/keeneland.capability.submit.sh @ONLY)
