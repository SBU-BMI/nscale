#!/bin/bash

transports="NULL na-NULL POSIX na-POSIX MPI MPI_AMR"
filecount=50
subgroup=2
iosize=4
buffersize=4
procs=8

ct_min=0
ct_max=0.01

bindir=/home/tcpan/PhD/path/src/nscale-debug

for transport in ${transports}
do


echo mpirun -np ${procs} ${bindir}/bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-test/test -P ${iosize} -t ${transport} -p ${subgroup} -n ${filecount} -d ${ct_min} -D ${ct_max}
mpirun -np ${procs} ${bindir}/bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-test/test -P ${iosize} -t ${transport} -p ${subgroup} -n ${filecount} -d ${ct_min} -D ${ct_max}

echo mpirun -np ${procs} ${bindir}/bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-test/test -P ${iosize} -t ${transport} -p ${subgroup} -n ${filecount} -d ${ct_min} -D ${ct_max}
mpirun -np ${procs} ${bindir}/bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-test/test -P ${iosize} -t ${transport} -p ${subgroup} -n ${filecount} -d ${ct_min} -D ${ct_max}


done
