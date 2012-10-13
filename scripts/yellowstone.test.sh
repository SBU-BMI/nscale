#!/bin/bash

mpirun -np 8 bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/synpush-sep-4.4.2-1.2-1-na-POSIX 5 cpu na-POSIX 4 2 1 0 4096 on
mpirun -np 8 bin/SynData_Push.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/synpush-sep-4.4.2-1.2-1-na-POSIX 5 cpu na-POSIX 4 2 1 0 4096 off

mpirun -np 8 bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/synfull-sep-8.4.2-1.2-1-na-POSIX 30 cpu na-POSIX 4 2 1 2 1 0 4096 off
mpirun -np 8 bin/SynData_Full.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/synfull-sep-8.4.2-1.2-1-na-POSIX 30 cpu na-POSIX 4 2 1 2 1 0 4096 on

mpirun -np 8 bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/astro-sep-8.4.2-1.2-1-na-POSIX -1 cpu na-POSIX 4 2 1 2 1 0 on
mpirun -np 8 bin/Process_test.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/astro-sep-8.4.2-1.2-1-na-POSIX -1 cpu na-POSIX 4 2 1 2 1 0 off

mpirun -np 8 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/baseline-8.4.2-1.2-1-MPI -1 0-200 cpu 1 off
mpirun -np 8 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/baseline-8.4.2-1.2-1-MPI -1 0-200 cpu 1 on

mpirun -np 8 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/adios-coloc-8.4.2-1-MPI MPI -1 4 cpu 2 1 0 off
mpirun -np 8 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/adios-coloc-8.4.2-1-MPI MPI -1 4 cpu 2 1 0 on
