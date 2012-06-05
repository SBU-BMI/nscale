#!/bin/sh
#PBS -N validation-adios
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
cd /lustre/medusa/tcpan/output

echo "nodefile="
cat $PBS_NODEFILE
echo "=end nodefile"

# run the program
mkdir /lustre/medusa/tcpan/output/posix-validation
mkdir /lustre/medusa/tcpan/output/adios-validation

which mpirun
date
echo "==== WARM UP ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/posix-validation -1 0-200 cpu"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/posix-validation -1 0-200 cpu
date
echo "==== ORIG ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/posix-validation -1 0-200 cpu"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/posix-validation -1 0-200 cpu
date
echo "==== ADIOS NULL ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation NULL cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation NULL cpu -1 -1
date
echo "==== ADIOS POSIX ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation POSIX cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation POSIX cpu -1 -1
date
echo "==== ADIOS MPI ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI cpu -1 -1
date
echo "==== ADIOS MPI_LUSTRE ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI_LUSTRE cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI_LUSTRE cpu -1 -1
date
echo "==== ADIOS MPI_AMR ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI_AMR cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/ValidationSet /lustre/medusa/tcpan/output/adios-validation MPI_AMR cpu -1 -1
date

# eof
