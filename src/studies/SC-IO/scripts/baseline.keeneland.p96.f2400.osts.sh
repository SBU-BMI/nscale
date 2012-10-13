#!/bin/sh
#PBS -N TCGA.baseline.keeneland.p96.f2400.osts
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
#PBS -l nodes=8:ppn=12

### End of PBS options ###

source /nics/c/home/tcpan/keeneland_env.sh

BINDIR=/nics/c/home/tcpan/builds/nscale-keeneland-cpu
DATADIR=/lustre/medusa/tcpan/bcrTCGA_osts0_14
# old, no attention to OSTS.  DATADIR=/lustre/medusa/tcpan/bcrTCGA
OUTDIR=/lustre/medusa/tcpan/output/baseline

cd $OUTDIR

##cd $PBS_O_WORKDIR
echo "nodefile="
cat $PBS_NODEFILE
echo "=end nodefile"

date
echo "==== ORIG ===="
echo "mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/nu-segment-scio.exe ${DATADIR} ${OUTDIR}/TCGA.baseline.keeneland.p96.f2400 2400 0-200 cpu"
mpirun --mca mpi_paffinity_alone 1 ${BINDIR}/bin/nu-segment-scio.exe ${DATADIR} ${OUTDIR} 2400 0-200 cpu
date

mv ${OUTDIR}.csv ${OUTDIR}/../TCGA.baseline.keeneland.p96.f2400.osts.csv
exit 0
# eof
