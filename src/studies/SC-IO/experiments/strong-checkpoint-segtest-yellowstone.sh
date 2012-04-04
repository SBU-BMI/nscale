#!/bin/sh
DATA_DIR=/home/tcpan/PhD/path/Data
BIN_DIR=/home/tcpan/PhD/path/src/nscale-release/bin
SCRIPT_FILE=$0
OUT_DIR=${SCRIPT_FILE%.*}

echo "running $0"

mkdir -p  -v $OUT_DIR

# strong scaling test using segmentation test data (5 images) with all stages turned on, on yellowstone.
for i in 2 3 5 9
do
	echo "running strong scaling seg test on yellowstone with $i processes and checkpointing all"
	mpirun -np $i ${BIN_DIR}/nu-segment-scio.exe ${DATA_DIR}/segmentation-tests ${DATA_DIR}/scio-segtest-mpi 5 0-101 cpu 1 > ${OUT_DIR}/scio-segtest-mpi${i}.csv
done

echo "$0 completed"
