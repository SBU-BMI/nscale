#!/bin/sh
#PBS -N valid-tests
#PBS -j oe
#PBS -A UT-NTNL0111
#PBS -m abe
#PBS -M tcpan@emory.edu

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=04:00:00
##PBS -l nodes=12:ppn=4:gpus=3
#PBS -l nodes=12:ppn=12

### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

date
##cd $PBS_O_WORKDIR

echo "nodefile="
cat $PBS_NODEFILE
echo "=end nodefile"

# run the program

which mpirun
date
echo "==== POSIX ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/posix-val-test -1 0-200 cpu 1
date
echo "==== ADIOS MPI-IO ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-mpio-val-test valtest cpu 1
date

# eof
