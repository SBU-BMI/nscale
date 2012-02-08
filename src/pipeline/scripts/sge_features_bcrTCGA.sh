#!bin/bash

#$ -S /bin/bash
#$ -j y

#$ -cwd

#$ -V
echo "Got $NSLOTS processors."
# does not need host file really - openmpi work with sge to get the correct resources.  -hostfile /data/tcpan/hostfiles/allnodes 
mpirun -np $NSLOTS /data/tcpan/src/nscale/src/pipeline/nu-features.exe /data/tcpan2/output/bcrTCGA/20Xtiles /data2/Images/bcrTCGA/diagnostic_block_HE_section_image/20Xtiles /data/tcpan2/output/bcrTCGA/features bcrTCGA cpu 
