#!/bin/bash
IN_DATA_DIR=/data/exascale/DATA
OUT_DATA_DIR=/data/tcpan/output
BIN_DIR=/data/tcpan/src/nscale-bin/bin
HOSTFILE=/data/tcpan/hostfiles/gpunodes
FILE_COUNT=3200

SCRIPT_FILE=$0
OUT_DIR=${SCRIPT_FILE%.*}
OUT_PREFIX=$(basename $0 .sh)

echo "running $0"

mkdir -p -v $OUT_DIR

# strong scaling test using segmentation test data (5 images) with all stages turned on, on yellowstone.
for i in 129 65 33 17 9 5
do
	echo "running strong scaling with validation data on cci cluster with $((i-1)) worker processes and checkpointing all"
	mpirun -np $i --hostfile ${HOSTFILE} --bynode ${BIN_DIR}/nu-segment-scio.exe ${IN_DATA_DIR}/ValidationSet ${OUT_DATA_DIR}/scio-val-mpi $FILE_COUNT 100 cpu 1 &> ${OUT_DIR}/${OUT_PREFIX}.${i}.csv
done

echo "$0 completed"
