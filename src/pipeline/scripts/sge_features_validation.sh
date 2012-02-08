#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."
# does not need host file really - openmpi work with sge to get the correct resources.  -hostfile /data/tcpan/hostfiles/allnodes 
mpirun -np $NSLOTS /data/tcpan/src/nscale/src/pipeline/nu-features.exe /data/tcpan/output/validation/20X_4096x4096_tiles /data/exascale/DATA/ValidationSet/20X_4096x4096_tile /data/tcpan/output/features/20X_4096x4096_tiles validation cpu 
