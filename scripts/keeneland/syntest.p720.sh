#!/bin/sh
#PBS -N synthetic.datasizes.p720
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
#PBS -l nodes=50:ppn=12
### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

JOB_PREFIX="synthetic.datasizes.p720"

BINDIR=/nics/c/home/tcpan/builds/nscale-keeneland-cpu
DATADIR=/lustre/medusa/tcpan/bcrTCGA
# old, no attention to OSTS.  DATADIR=/lustre/medusa/tcpan/bcrTCGA
OUTDIR=/lustre/medusa/tcpan/output/${JOB_PREFIX}

cd $OUTDIR

buffer_sizes="2 4 6"
data_sizes="128 256 512 1024 2048 4096 8192"
io_sizes="15 30 60 120 240 360 480"
transports="na-NULL NULL na-POSIX POSIX MPI_AMR"

date

for buffer_size in ${buffer_sizes}
do
	for data_size in ${data_sizes}
	do

		for io_size in ${io_sizes}
		do
			for transport in ${transports}
			do
				date
				echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.b 6000 cpu ${transport} ${buffer_size} ${io_size} 1 15 ${data_size} off off"
				mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.b 6000 cpu ${transport} ${buffer_size} ${io_size} 1 15 ${data_size} off off
				rm -rf ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.b

				echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.nb 6000 cpu ${transport} ${buffer_size} ${io_size} 1 15 ${data_size} off on"
				mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.nb 6000 cpu ${transport} ${buffer_size} ${io_size} 1 15 ${data_size} off on

				rm -rf ${OUTDIR}/syntest.n50.f6000.${transport}.b${buffer_size}.io${io_size}.is15.data${data_size}.nb
			done
		done		
	done
done
date

exit 0
