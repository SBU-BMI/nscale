#!/bin/sh
#PBS -N tcga_small2-adios
#PBS -j oe
#PBS -A UT-NTNL0111
#PBS -m abe
#PBS -M tcpan@emory.edu

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=06:00:00
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
mkdir /lustre/medusa/tcpan/output/posix-tcga_small2
mkdir /lustre/medusa/tcpan/output/adios-tcga_small2-all
mkdir /lustre/medusa/tcpan/output/adios-tcga_small2-1
mkdir /lustre/medusa/tcpan/output/adios-tcga_small2-12a
mkdir /lustre/medusa/tcpan/output/adios-tcga_small2-12i


which mpirun
date
echo "==== ORIG ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/posix-tcga_small2 -1 0-200 cpu"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/posix-tcga_small2 -1 0-200 cpu
date
echo "**** 1 group of everyone, 1 proc per group, 24 groups of 12 adjacent, 24 groups of 12 interleaved ****"
echo "==== ADIOS NULL ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all NULL cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all NULL cpu -1 -1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 NULL cpu 1 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 NULL cpu 1 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a NULL cpu 12 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a NULL cpu 12 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i NULL cpu 12 24"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i NULL cpu 12 24
date
echo "==== ADIOS MPI_LUSTRE ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI_LUSTRE cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI_LUSTRE cpu -1 -1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI_LUSTRE cpu 1 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI_LUSTRE cpu 1 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI_LUSTRE cpu 12 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI_LUSTRE cpu 12 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI_LUSTRE cpu 12 24"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI_LUSTRE cpu 12 24
date
echo "==== ADIOS MPI_AMR ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI_AMR cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI_AMR cpu -1 -1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI_AMR cpu 1 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI_AMR cpu 1 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI_AMR cpu 12 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI_AMR cpu 12 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI_AMR cpu 12 24"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI_AMR cpu 12 24
date
echo "==== ADIOS POSIX ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all POSIX cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all POSIX cpu -1 -1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 POSIX cpu 1 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 POSIX cpu 1 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a POSIX cpu 12 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a POSIX cpu 12 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i POSIX cpu 12 24"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i POSIX cpu 12 24
date
echo "==== ADIOS MPI ===="
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI cpu -1 -1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-all MPI cpu -1 -1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI cpu 1 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-1 MPI cpu 1 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI cpu 12 1"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12a MPI cpu 12 1
echo "mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI cpu 12 24"
mpirun --mca mpi_paffinity_alone 1 /nics/c/home/tcpan/builds/nscale-keeneland-cpu/bin/nu-segment-scio-adios.exe /lustre/medusa/tcpan/bcrTCGA_small2 /lustre/medusa/tcpan/output/adios-tcga_small2-12i MPI cpu 12 24
date

# eof
