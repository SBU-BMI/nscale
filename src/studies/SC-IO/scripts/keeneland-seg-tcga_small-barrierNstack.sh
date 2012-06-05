#!/bin/sh
#PBS -N tcga_small-adios
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
#PBS -l nodes=24:ppn=12

### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

date
##cd $PBS_O_WORKDIR
cd /lustre/medusa/tcpan/output

echo "nodefile="
cat $PBS_NODEFILE
echo "=end nodefile"

# run the program
mkdir /lustre/medusa/tcpan/output/posix-tcga_small
mkdir /lustre/medusa/tcpan/output/adios-tcga_small
mkdir /lustre/medusa/tcpan/output/adios-tcga_small-stack


which mpirun
date
echo "==== ORIG ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/posix-tcga_small -1 0-200 cpu"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/posix-tcga_small -1 0-200 cpu
date
echo "==== ADIOS NULL ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small NULL cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small NULL cpu -1 -1
date
echo "==== ADIOS NULL stack ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-NULL cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-NULL cpu -1 -1
date
echo "==== ADIOS POSIX ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small POSIX cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small POSIX cpu -1 -1
date
echo "==== ADIOS POSIX stack ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-POSIX cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-POSIX cpu -1 -1
date
echo "==== ADIOS MPI ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI cpu -1 -1
date
echo "==== ADIOS MPI stack ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI cpu -1 -1
date
echo "==== ADIOS MPI_LUSTRE ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_LUSTRE cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_LUSTRE cpu -1 -1
date
echo "==== ADIOS MPI_LUSTRE stack ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI_LUSTRE cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI_LUSTRE cpu -1 -1
date
echo "==== ADIOS MPI_AMR ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_AMR cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small MPI_AMR cpu -1 -1
date
echo "==== ADIOS MPI_AMR stack ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI_AMR cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small /lustre/medusa/tcpan/output/adios-tcga_small-stack gap-MPI_AMR cpu -1 -1
date

# eof
