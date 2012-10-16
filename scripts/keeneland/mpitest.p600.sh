#!/bin/sh
#PBS -N mpitests.p600
#PBS -j oe
#PBS -A UT-NTNL0111
#PBS -m abe
#PBS -M tcpan@emory.edu

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=01:00:00
##PBS -l nodes=12:ppn=4:gpus=3
#PBS -l nodes=50:ppn=12
### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

BINDIR=/nics/c/home/tcpan/builds/nscale-keeneland-cpu
DATADIR=/lustre/medusa/tcpan/bcrTCGA
# old, no attention to OSTS.  DATADIR=/lustre/medusa/tcpan/bcrTCGA
OUTDIR=/lustre/medusa/tcpan/output

cd $OUTDIR

sizes="128 256 512 1024 2048 4096 8192"

	for size in ${sizes}
	do
		date
		echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/test-mpi-iprobe-orders.exe 60 5 ${size}"
		mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/test-mpi-iprobe-orders.exe 60 5 ${size} 
	done

	for size in ${sizes}
	do
		date
		echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/test-mpi-send-delays.exe 60 15 5 4 ${size}"
		mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/test-mpi-send-delays.exe 60 15 5 4 ${size}

		mv mpi_datasize_test.csv mpi-send-delays.60-15-5-4-${size}.csv 
	done


	


exit 0
