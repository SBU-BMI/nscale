#!/bin/bash

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed-NB/synpush-sep-128.4.32-1.8-1-na-POSIX 5 cpu na-POSIX 4 32 1 8 1 4096 on on
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed-NB/synpush-sep-128.4.32-1.8-1-na-POSIX 5 cpu na-POSIX 4 32 1 8 1 4096 off on

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed/synpush-sep-128.4.32-1.8-1-na-POSIX 5 cpu na-POSIX 4 32 1 8 1 4096 on off
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed/synpush-sep-128.4.32-1.8-1-na-POSIX 5 cpu na-POSIX 4 32 1 8 1 4096 off off

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed-NB/synfull-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 4096 off on
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed-NB/synfull-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 4096 on on

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed/synfull-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 4096 off off
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed/synfull-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 4096 on off

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed-NB/astro-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 on on
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed-NB/astro-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 off on

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed/astro-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 on off
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed/astro-sep-128.4.32-1.8-1-na-POSIX -1 cpu na-POSIX 4 32 1 8 1 off off

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed/baseline-128-MPI -1 0-200 cpu 1 off
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed/baseline-128-MPI -1 0-200 cpu 1 on

mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-uncompressed/adios-coloc-128.4.8-1-MPI MPI -1 4 cpu 8 1 0 off
mpirun -np 128 --bynode -hostfile /data/tcpan/hostfiles/non_nfs_nodes bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles /mnt/scratch1/tcpan/adios/cciclus-preDB-compressed/adios-coloc-128.4.8-1-MPI MPI -1 4 cpu 8 1 0 on

