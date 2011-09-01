#!/bin/sh
#PBS -N feature-3-4-5
#PBS -j oe
#PBS -A UT-NTNL0111

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=05:05:00
#PBS -l nodes=1:ppn=3:gpu=3

### End of PBS options ###

date
cd $PBS_O_WORKDIR

./run3.sh

# run the program


# eof
