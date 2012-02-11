#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."
# does not need host file really - openmpi work with sge to get the correct resources.  -hostfile /data/tcpan/hostfiles/allnodes 
mpirun -np $NSLOTS /data/tcpan/src/nscale/src/pipeline/nu-features-file-upgrade.exe /data/tcpan2/output/bcrTCGA/features bcrTCGA cpu 
