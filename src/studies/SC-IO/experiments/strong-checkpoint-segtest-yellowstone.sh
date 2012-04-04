#!/bin/sh
IN_DATA_DIR=/home/tcpan/PhD/path/Data
OUT_DATA_DIR=/home/tcpan/PhD/path/Data
BIN_DIR=/home/tcpan/PhD/path/src/nscale-release/bin
FILE_COUNT=5

SCRIPT_FILE=$0
OUT_DIR=${SCRIPT_FILE%.*}
OUT_PREFIX=$(basename $0 .sh)

echo "running $0"

mkdir -p -v $OUT_DIR

# strong scaling test using segmentation test data (5 images) with all stages turned on, on yellowstone.
for i in 2 3 5 9
do
	echo "running strong scaling seg test on yellowstone with $((i-1)) worker processes and checkpointing all"
	mpirun -np $i ${BIN_DIR}/nu-segment-scio.exe ${IN_DATA_DIR}/segmentation-tests ${OUT_DATA_DIR}/scio-segtest-mpi ${FILE_COUNT} 0-101 cpu 1 > ${OUT_DIR}/${OUT_PREFIX}.${i}.csv
done

echo "$0 completed"
