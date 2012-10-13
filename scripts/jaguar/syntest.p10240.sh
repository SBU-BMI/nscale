#!/bin/bash
#    Begin PBS directives
#PBS -A csc025ewk
#PBS -M tcpan@emory.edu
#PBS -m abe
#PBS -N synthetic.datasizes.p10240
#PBS -j oe
#PBS -l walltime=06:00:00,size=10240
#PBS -l gres=widow2%widow3
#PBS -V
#    End PBS directives and begin shell commands

# replaced with -V.   source ~/jaguar_env.sh

BINDIR=/tmp/work/pantc/nscale
DATADIR=/lustre/widow2/proj/csc025/tkurc1/bcrTCGA-new
# old, no attention to OSTS.  DATADIR=/lustre/widow2/proj/csc025/tkurc1/bcrTCGA/20Xtiles
OUTDIR=/lustre/widow2/proj/csc025/pantc

cd $OUTDIR
sizes="128 256 512 1024 2048 4096 8192"
transports="na-NULL NULL na-POSIX POSIX MPI_AMR"

for transport in ${transports}
do
	for size in ${sizes}
	do
		date
		echo "aprun -S 8 -n 10240 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/synthetic.datasizes.p10240 100000 cpu ${transport} 4 600 1 15 1 0 ${size}"
		aprun -S 8 -n 10240 ${BINDIR}/bin/SynData_Full.exe ${DATADIR} ${OUTDIR}/synthetic.datasizes.p10240 100000 cpu ${transport} 4 600 1 15 1 0 ${size} 

		rm -rf ${OUTDIR}/synthetic.datasizes.p10240
        mv ${OUTDIR}/synthetic.datasizes.p10240.csv ${OUTDIR}/synthetic.datasizes.p10240.${transport}.${size}.csv
	done
done

exit 0
