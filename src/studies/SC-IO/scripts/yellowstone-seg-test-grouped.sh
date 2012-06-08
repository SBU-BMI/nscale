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
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2
mkdir /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2

which mpirun
date
echo "==== ORIG ===="
mpirun -np 5 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/posix-seg-tests2 -1 0-200 cpu 1
date
echo "==== ADIOS GROUP TEST ===="
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 NULL cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 POSIX cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 gap-NULL cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 gap-NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 gap-NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 gap-NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 gap-NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 gap-NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 gap-MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 gap-MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 gap-MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 gap-MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 gap-MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 gap-MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa-2 gap-POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11-2 gap-POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21-2 gap-POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31-2 gap-POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22-2 gap-POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23-2 gap-POSIX cpu 2 3
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
