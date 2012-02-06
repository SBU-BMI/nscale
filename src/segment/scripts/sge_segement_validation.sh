#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."

mpirun -np $NSLOTS -hostfile /data/tcpan/hostfiles/allnodes /data/tcpan/src/nscale/src/segment/nu-segment.exe /data/exascale/DATA/ValidationSet/20X_4096x4096_tiles /data/tcpan/output/validation test cpu 
