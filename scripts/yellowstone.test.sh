#!/bin/bash

mpirun -np 8 bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed-NB/synpush-sep-8.4.2-1.2-1-na-POSIX -n 40 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 1 -l 1
mpirun -np 8 bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed-NB/synpush-sep-8.4.2-1.2-1-na-POSIX -n 40 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 0 -l 1
                                                                                                           
mpirun -np 8 bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/synpush-sep-8.4.2-1.2-1-na-POSIX -n 40 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 1 -l 0
mpirun -np 8 bin/SynData_Push.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/synpush-sep-8.4.2-1.2-1-na-POSIX -n 40 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 0 -l 0
                                                                                                           
mpirun -np 8 bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed-NB/synfull-sep-8.4.2-1.2-1-na-POSIX -n 30 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 1 -l 1
mpirun -np 8 bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed-NB/synfull-sep-8.4.2-1.2-1-na-POSIX -n 30 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 0 -l 1
                                                                                                           
mpirun -np 8 bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/synfull-sep-8.4.2-1.2-1-na-POSIX -n 30 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 1 -l 0
mpirun -np 8 bin/SynData_Full.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/synfull-sep-8.4.2-1.2-1-na-POSIX -n 30 -t na-POSIX -b 4 -P 2 -p 2 -m 4096 -c 0 -l 0

mpirun -np 8 bin/SegmentNuclei.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed-NB/astro-sep-8.4.2-1.2-1-na-POSIX -n -1 -t na-POSIX -b 4 -P 2 -p 2 -c 1 -l 1
mpirun -np 8 bin/SegmentNuclei.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed-NB/astro-sep-8.4.2-1.2-1-na-POSIX -n -1 -t na-POSIX -b 4 -P 2 -p 2 -c 0 -l 1
                                                                                                            
mpirun -np 8 bin/SegmentNuclei.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/astro-sep-8.4.2-1.2-1-na-POSIX -n -1 -t na-POSIX -b 4 -P 2 -p 2 -c 1 -l 0
mpirun -np 8 bin/SegmentNuclei.exe -i /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 -o /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/astro-sep-8.4.2-1.2-1-na-POSIX -n -1 -t na-POSIX -b 4 -P 2 -p 2 -c 0 -l 0

mpirun -np 8 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/baseline-8-MPI -1 0-200 cpu 1 off
mpirun -np 8 bin/nu-segment-scio.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/baseline-8-MPI -1 0-200 cpu 1 on

mpirun -np 8 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-uncompressed/adios-coloc-8.4.2-1-MPI MPI -1 4 cpu 2 1 0 off
mpirun -np 8 bin/nu-segment-scio-adios.exe /home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1 /home/tcpan/PhD/path/Data/adios/yellowstone-preDB-compressed/adios-coloc-8.4.2-1-MPI MPI -1 4 cpu 2 1 0 on
