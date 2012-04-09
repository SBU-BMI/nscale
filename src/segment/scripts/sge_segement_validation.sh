#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."

mpirun -np $NSLOTS /data/tcpan/src/nscale-bin/bin/nu-segment.exe /data/exascale/DATA/ValidationSet/20X_4096x4096_tiles /data/tcpan/output/validation test cpu 

