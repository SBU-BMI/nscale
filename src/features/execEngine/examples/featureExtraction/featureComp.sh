#!/bin/sh
#PBS -N features-Multicore
#PBS -j oe
#PBS -A UT-NTNL0111

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=04:10:00
#PBS -l nodes=1:ppn=12:gpus=1

### End of PBS options ###

date
cd $PBS_O_WORKDIR

./run.sh

echo "done"
date
# run the program


# eof
