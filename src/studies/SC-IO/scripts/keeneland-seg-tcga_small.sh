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
echo "==== WARM UP ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/posix-tcga_small -1 0-200 cpu 1
date
echo "==== ORIG ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/posix-tcga_small -1 0-200 cpu 1
date
echo "==== ADIOS NULL ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small NULL cpu 1
date
echo "==== ADIOS POSIX ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small POSIX cpu 1
date
echo "==== ADIOS MPI ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI cpu 1
date
echo "==== ADIOS MPI_LUSTRE ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_LUSTRE cpu 1
date
echo "==== ADIOS MPI_AMR ===="
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_AMR cpu 1
date

# eof
