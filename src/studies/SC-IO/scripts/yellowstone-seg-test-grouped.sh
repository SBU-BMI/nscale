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


date
##cd $PBS_O_WORKDIR

# run the program

which mpirun
date
echo "==== ORIG ===="
mpirun -np 5 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/posix-seg-tests -1 0-200 cpu 1
date
echo "==== ADIOS GROUP TEST ===="
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa NULL cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a NULL cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a NULL cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a NULL cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a NULL cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 NULL cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 NULL cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 NULL cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 NULL cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 NULL cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 NULL cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 NULL cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 NULL cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 NULL cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 NULL cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 NULL cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 NULL cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 NULL cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 NULL cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 NULL cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 NULL cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 NULL cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 NULL cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 NULL cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 NULL cpu 4 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a MPI cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a MPI cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a MPI cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a MPI cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 MPI cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 MPI cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 MPI cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 MPI cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 MPI cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 MPI cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 MPI cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 MPI cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 MPI cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 MPI cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 MPI cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 MPI cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 MPI cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 MPI cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 MPI cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 MPI cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 MPI cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 MPI cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 MPI cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 MPI cpu 4 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a POSIX cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a POSIX cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a POSIX cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a POSIX cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 POSIX cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 POSIX cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 POSIX cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 POSIX cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 POSIX cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 POSIX cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 POSIX cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 POSIX cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 POSIX cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 POSIX cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 POSIX cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 POSIX cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 POSIX cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 POSIX cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 POSIX cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 POSIX cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 POSIX cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 POSIX cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 POSIX cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 POSIX cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 POSIX cpu 4 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-NULL cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a gap-NULL cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a gap-NULL cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a gap-NULL cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a gap-NULL cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 gap-NULL cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 gap-NULL cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 gap-NULL cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 gap-NULL cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 gap-NULL cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 gap-NULL cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 gap-NULL cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 gap-NULL cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 gap-NULL cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 gap-NULL cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 gap-NULL cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 gap-NULL cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 gap-NULL cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 gap-NULL cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 gap-NULL cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 gap-NULL cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 gap-NULL cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 gap-NULL cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 gap-NULL cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 gap-NULL cpu 4 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a gap-MPI cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a gap-MPI cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a gap-MPI cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a gap-MPI cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 gap-MPI cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 gap-MPI cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 gap-MPI cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 gap-MPI cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 gap-MPI cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 gap-MPI cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 gap-MPI cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 gap-MPI cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 gap-MPI cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 gap-MPI cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 gap-MPI cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 gap-MPI cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 gap-MPI cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 gap-MPI cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 gap-MPI cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 gap-MPI cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 gap-MPI cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 gap-MPI cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 gap-MPI cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 gap-MPI cpu 4 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-1a gap-POSIX cpu 1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-2a gap-POSIX cpu 2 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-3a gap-POSIX cpu 3 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-4a gap-POSIX cpu 4 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a1 gap-POSIX cpu -1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-41 gap-POSIX cpu 4 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a2 gap-POSIX cpu -1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-12 gap-POSIX cpu 1 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-32 gap-POSIX cpu 3 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-42 gap-POSIX cpu 4 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a3 gap-POSIX cpu -1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-13 gap-POSIX cpu 1 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-POSIX cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-33 gap-POSIX cpu 3 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-43 gap-POSIX cpu 4 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a4 gap-POSIX cpu -1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-14 gap-POSIX cpu 1 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-24 gap-POSIX cpu 2 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-34 gap-POSIX cpu 3 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-44 gap-POSIX cpu 4 4
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-a5 gap-POSIX cpu -1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-15 gap-POSIX cpu 1 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-25 gap-POSIX cpu 2 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-35 gap-POSIX cpu 3 5
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-45 gap-POSIX cpu 4 5
date
#echo "==== ADIOS NULL ===="
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu -1 -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu -1 -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu 2 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu 2 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu 2 2
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu 2 2
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu 3 2
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu 3 2
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu 8 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu 8 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests NULL cpu 1 -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-NULL cpu 1 -1
#date
#echo "==== ADIOS POSIX ===="
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests POSIX cpu -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-POSIX cpu -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests POSIX cpu 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-POSIX cpu 1
#date
#echo "==== ADIOS MPI ===="
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests MPI cpu -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-MPI cpu -1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests MPI cpu 1
#mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios-seg-tests gap-MPI cpu 1
#date

# eof
