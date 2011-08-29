#!/bin/sh
#PBS -N featuresComp-3-4-5
#PBS -j oe
#PBS -A UT-NTNL0111

### Unused PBS options ###
##
#PBS -l walltime=06:05:00
#PBS -l nodes=1:ppn=3

### End of PBS options ###

date
cd $PBS_O_WORKDIR

./run2.sh

# run the program


# eof
