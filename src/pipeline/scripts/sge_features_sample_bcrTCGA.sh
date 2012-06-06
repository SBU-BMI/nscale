#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."
# does not need host file really - openmpi work with sge to get the correct resources.  -hostfile /data/tcpan/hostfiles/allnodes 
mpirun -np $NSLOTS /data/tcpan/src/nscale-bin/bin/nu-features-sample.exe /data/tcpan2/output/bcrTCGA/features-green-by-image /data/tcpan2/output/bcrTCGA/features-green-sample-0.01 0.01 bcrTCGA0.01 cpu 
