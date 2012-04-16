#!/bin/sh
IN_DATA_DIR=/data/exascale/DATA
OUT_DATA_DIR=/data/tcpan/output
BIN_DIR=/data/tcpan/src/nscale-bin/bin
HOSTFILE=/data/tcpan/hostfiles/gpunodes
FILE_COUNT=25

SCRIPT_FILE=$0
OUT_DIR=${SCRIPT_FILE%.*}
OUT_PREFIX=$(basename $0 .sh)

echo "running $0"

mkdir -p -v $OUT_DIR

# strong scaling test using segmentation test data (5 images) with all stages turned on, on yellowstone.
for i in 5 9 17 33 65 129
do
	echo "running weak scaling with validation data on cci cluster with $((i-1)) worker processes and checkpointing all"
	TOTAL_COUNT=$((i-1))
	TOTAL_COUNT=$((TOTAL_COUNT * FILE_COUNT))
	mpirun -np $i --hostfile ${HOSTFILE} --bynode ${BIN_DIR}/nu-segment-scio.exe ${IN_DATA_DIR}/ValidationSet ${OUT_DATA_DIR}/scio-val-mpi ${TOTAL_COUNT} 0-101 cpu 1 > ${OUT_DIR}/${OUT_PREFIX}.${i}.csv
done

echo "$0 completed"
