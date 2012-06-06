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
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 POSIX cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-NULL cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-NULL cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-NULL cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-NULL cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-NULL cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-NULL cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-MPI cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-MPI cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-MPI cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-MPI cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-MPI cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-MPI cpu 2 3
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-aa gap-POSIX cpu -1 -1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-11 gap-POSIX cpu 1 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-21 gap-POSIX cpu 2 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-31 gap-POSIX cpu 3 1
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-22 gap-POSIX cpu 2 2
mpirun -np 5 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/segmentation-tests /home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-23 gap-POSIX cpu 2 3
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
