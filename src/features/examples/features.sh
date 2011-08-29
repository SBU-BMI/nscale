#!/bin/sh
#PBS -N features
#PBS -j oe
#PBS -A UT-NTNL0111

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=00:01:00
#PBS -l nodes=1:ppn=1

### End of PBS options ###

date
cd $PBS_O_WORKDIR

echo "nodefile="
cat $PBS_NODEFILE
echo "=end nodefile"

# run the program

./testRegional ../../old/FeaturesLib/images/manyNucleiMask.png ../../old/FeaturesLib/images/manyNucleiGray.png

date

# eof
