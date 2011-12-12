#!/bin/sh
#PBS -N features-2GP
#PBS -j oe
#PBS -A UT-NTNL0111

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=01:10:00
#PBS -l nodes=1:ppn=2:gpus=3

### End of PBS options ###

date
cd $PBS_O_WORKDIR

./runMultiGPU.sh

# run the program


# eof
