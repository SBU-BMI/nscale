#!/bin/sh
#PBS -N synthetic.datasizes.p1200
#PBS -j oe
#PBS -A UT-NTNL0111
#PBS -m abe
#PBS -M tcpan@emory.edu

### Unused PBS options ###
## If left commented, must be specified when the job is submitted:
## 'qsub -l walltime=hh:mm:ss,nodes=12:ppn=4'
##
#PBS -l walltime=06:00:00
##PBS -l nodes=12:ppn=4:gpus=3
#PBS -l nodes=100:ppn=12
### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

BINDIR=/nics/c/home/tcpan/builds/nscale-keeneland-cpu
DATADIR=/lustre/medusa/tcpan/bcrTCGA
# old, no attention to OSTS.  DATADIR=/lustre/medusa/tcpan/bcrTCGA
OUTDIR=/lustre/medusa/tcpan/output

cd $OUTDIR

sizes="128 256 512 1024 2048 4096 8192"
transports="na-NULL NULL na-POSIX POSIX MPI_AMR"

for transport in ${transports}
do
	for size in ${sizes}
	do
		date
		echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe -i ${DATADIR} -o ${OUTDIR}/synthetic.datasizes.p1200 -n 30000 -t ${transport} -b 4 -P 60 -p 15 -m ${size} -c 0 -l 1" 
		mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/SynData_Full.exe -i ${DATADIR} -o ${OUTDIR}/synthetic.datasizes.p1200 -n 30000 -t ${transport} -b 4 -P 60 -p 15 -m ${size} -c 0 -l 1 
		
		
		rm -rf ${OUTDIR}/synthetic.datasizes.p1200
        mv ${OUTDIR}/synthetic.datasizes.p1200.csv ${OUTDIR}/synthetic.datasizes.p1200.${transport}.${size}.csv
		
	done
done

exit 0
