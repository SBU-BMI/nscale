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

mkdir /data/tcpan/output/adios-val-aa
mkdir /data/tcpan/output/adios-val-11
mkdir /data/tcpan/output/adios-val-81
mkdir /data/tcpan/output/adios-val-86


# run the program

which mpirun
date
echo "==== ORIG ===="
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/posix-val -1 0-200 cpu 1
date
echo "==== ADIOS GROUP TEST ===="
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-aa NULL cpu -1 -1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 NULL cpu 1 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-81 NULL cpu 8 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-86 NULL cpu 8 6
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-aa POSIX cpu -1 -1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 POSIX cpu 1 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-81 POSIX cpu 8 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-86 POSIX cpu 8 6
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-aa gap-NULL cpu -1 -1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 gap-NULL cpu 1 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-81 gap-NULL cpu 8 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-86 gap-NULL cpu 8 6
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-aa gap-POSIX cpu -1 -1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 gap-POSIX cpu 1 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-81 gap-POSIX cpu 8 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-86 gap-POSIX cpu 8 6
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 MPI cpu 1 1
mpirun -np 48 -hostfile /data/tcpan/hostfiles/oldnodes --bynode /data/tcpan/src/nscale-bin/bin/nu-segment-scio-adios.exe /data/exascale/DATA/ValidationSet /data/tcpan/output/adios-val-11 gap-MPI cpu 1 1


# eof
