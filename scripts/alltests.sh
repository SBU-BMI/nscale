#!/bin/bash

bindir=/home/tcpan/PhD/path/src/nscale-release

#below stuff works.
echo FileUtils test
${bindir}/bin/FileUtilsTest.exe
echo FileUtils test done.

echo DataBuffer test	
${bindir}/bin/DataBuffer_test.exe
mpirun -n 8 ${bindir}/bin/DataBuffer_test.exe
echo DataBuffer test done.

echo Scheduler Test
mpirun -np 8 ${bindir}/bin/Scheduler_test.exe
echo Scheduler Test done.

echo Dummy Assign test
${bindir}/bin/Assign_test.exe
echo Dummy Assign test done.

echo Dummy Segment test
${bindir}/bin/Segment_test.exe
echo Dummy Segment test done

echo Dummy Save test
mpirun -np 8 ${bindir}/bin/Save_test.exe
echo Dummy Save test done

echo PullHandler test
mpirun -np 8 ${bindir}/bin/PullHandler_test.exe
echo PullHandler test done.

echo PushHandler test
mpirun -n 8 ${bindir}/bin/PushHandler_test.exe
echo PushHandler test done.

echo CVImage test
${bindir}/bin/CVImage_test.exe
echo CVImage test done.

echo Test CPP
${bindir}/bin/test-cpp.exe
echo Test CPP done.

echo Test MPI
mpirun -np 8 ${bindir}/bin/test-mpi.exe 4 2
mpirun -np 8 ${bindir}/bin/test-mpi-iprobe-orders.exe 4 20 4096 off
mpirun -np 8 ${bindir}/bin/test-mpi-iprobe-orders.exe 4 20 4096 on
mpirun -np 8 ${bindir}/bin/test-mpi-send-delays.exe 6 2 30 3 4096
echo Test MPI done.

echo Test ADIOS
cwd = `pwd`
cd $bindir; mpirun -np 8 ${bindir}/bin/test-adios.exe
cd ${cwd}
echo Test ADIOS done.

filecount=10
subgroup=2
iosize=4
buffersize=4
procs=8
datasize=4096
synComputeParams="0.34,0.17,0.007:0.05,3.6,0.125:0.61,11.5,1.7"

indir=/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1
outroot=/home/tcpan/PhD/path/Data/yellowstone/test


transports="POSIX na-POSIX NULL na-NULL MPI"
iosize=2
readsize=2

for transport in ${transports}
do
	echo ADIOS Separated Write - WORKS                                                                                                           
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-compressed-NB   -i ${indir} -o ${outroot}-sep-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-compressed-NB   -i ${indir} -o ${outroot}-sep-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 0
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-sep-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 0                                                                                                       
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-sep-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 0                                                                                                       
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-compressed      -i ${indir} -o ${outroot}-sep-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-compressed      -i ${indir} -o ${outroot}-sep-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 0
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-uncompressed    -i ${indir} -o ${outroot}-sep-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-sep-${transport}-uncompressed    -i ${indir} -o ${outroot}-sep-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 0
	
	echo ADIOS Separated Write Syn                                                                                                           
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-compressed-NB   -i ${indir} -o ${outroot}-synsep-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-compressed-NB   -i ${indir} -o ${outroot}-synsep-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-synsep-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}                                                                                                      
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-synsep-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}                                                                                                      
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-compressed      -i ${indir} -o ${outroot}-synsep-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-compressed      -i ${indir} -o ${outroot}-synsep-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	echo mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-uncompressed    -i ${indir} -o ${outroot}-synsep-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNuclei.exe -g ${outroot}-synsep-${transport}-uncompressed    -i ${indir} -o ${outroot}-synsep-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
done

for transport in ${transports}
do
	echo ADIOS Separated RW                                                                                                           
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-compressed-NB   -i ${indir} -o ${outroot}-sepR-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-compressed-NB   -i ${indir} -o ${outroot}-sepR-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 1 -l 1 -S 0
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-sepR-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 0                                                                                                       
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-sepR-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 0 -l 1 -S 0                                                                                                       
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-compressed      -i ${indir} -o ${outroot}-sepR-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-compressed      -i ${indir} -o ${outroot}-sepR-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 1 -l 0 -S 0
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-uncompressed    -i ${indir} -o ${outroot}-sepR-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 0
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-sepR-${transport}-uncompressed    -i ${indir} -o ${outroot}-sepR-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} -P ${iosize} -p ${subgroup} -c 0 -l 0 -S 0
	                                        
	echo ADIOS Separated RW Syn                                                                                                           
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-compressed-NB   -i ${indir} -o ${outroot}-synsepR-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 1 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-compressed-NB   -i ${indir} -o ${outroot}-synsepR-${transport}-compressed-NB   -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 1 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-synsepR-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 0 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}                                                                                                      
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-uncompressed-NB -i ${indir} -o ${outroot}-synsepR-${transport}-uncompressed-NB -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 0 -l 1 -S 1 -d ${synComputeParams} -m ${datasize}                                                                                                      
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-compressed      -i ${indir} -o ${outroot}-synsepR-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 1 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-compressed      -i ${indir} -o ${outroot}-synsepR-${transport}-compressed      -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 1 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	echo mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-uncompressed    -i ${indir} -o ${outroot}-synsepR-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 0 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}
	mpirun -np 8 ${bindir}/bin/SegmentNucleiReader.exe -g ${outroot}-synsepR-${transport}-uncompressed    -i ${indir} -o ${outroot}-synsepR-${transport}-uncompressed    -n ${filecount} -t ${transport} -b ${buffersize} -Q ${readsize} P ${iosize} -p ${subgroup} -c 0 -l 0 -S 1 -d ${synComputeParams} -m ${datasize}

done


echo Test Baseline - WORKS
echo mpirun -np 8 ${bindir}/bin/nu-segment-scio.exe ${indir} ${outroot}-baseline-uncompressed ${filecount} 0-200 cpu 1 off
mpirun -np 8 ${bindir}/bin/nu-segment-scio.exe ${indir} ${outroot}-baseline-uncompressed ${filecount} 0-200 cpu 1 off
echo mpirun -np 8 ${bindir}/bin/nu-segment-scio.exe ${indir} ${outroot}-baseline-compressed   ${filecount} 0-200 cpu 1 on
mpirun -np 8 ${bindir}/bin/nu-segment-scio.exe ${indir} ${outroot}-baseline-compressed   ${filecount} 0-200 cpu 1 on
echo Test Baseline Done.

echo Test Baseline Synthetic - WORKS
echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-synthetic.exe ${indir} ${outroot}-baseline-uncompressed-syn ${synComputeParams} ${filecount} 0-200 off
mpirun -np 8 ${bindir}/bin/nu-segment-scio-synthetic.exe ${indir} ${outroot}-baseline-uncompressed-syn ${synComputeParams} ${filecount} 0-200 off
echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-synthetic.exe ${indir} ${outroot}-baseline-compressed-syn   ${synComputeParams} ${filecount} 0-200 on
mpirun -np 8 ${bindir}/bin/nu-segment-scio-synthetic.exe ${indir} ${outroot}-baseline-compressed-syn   ${synComputeParams} ${filecount} 0-200 on
echo Test Baseline Synthetic DONE.

transports="POSIX NULL MPI"
for transport in ${transports}
do
	echo ADIOS Coloc - WORKS
	echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios.exe ${indir} ${outroot}-coloc-${transport}-uncompressed ${transport} ${filecount} ${buffersize} cpu ${subgroup} 1 0 off
	mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios.exe ${indir} ${outroot}-coloc-${transport}-uncompressed ${transport} ${filecount} ${buffersize} cpu ${subgroup} 1 0 off
	echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios.exe ${indir} ${outroot}-coloc-${transport}-compressed   ${transport} ${filecount} ${buffersize} cpu ${subgroup} 1 0 on
	mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios.exe ${indir} ${outroot}-coloc-${transport}-compressed   ${transport} ${filecount} ${buffersize} cpu ${subgroup} 1 0 on

	echo ADIOS Coloc Synth- WORKS
	echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios-synthetic.exe ${indir} ${outroot}-coloc-${transport}-uncompressed-syn ${transport} ${synComputeParams} ${filecount} ${buffersize} ${subgroup} 1 off
	mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios-synthetic.exe ${indir} ${outroot}-coloc-${transport}-uncompressed-syn ${transport} ${synComputeParams} ${filecount} ${buffersize} ${subgroup} 1 off
	echo mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios-synthetic.exe ${indir} ${outroot}-coloc-${transport}-compressed-syn   ${transport} ${synComputeParams} ${filecount} ${buffersize} ${subgroup} 1 on
	mpirun -np 8 ${bindir}/bin/nu-segment-scio-adios-synthetic.exe ${indir} ${outroot}-coloc-${transport}-compressed-syn   ${transport} ${synComputeParams} ${filecount} ${buffersize} ${subgroup} 1 on

done


#transports="POSIX na-POSIX NULL na-NULL MPI"
#for transport in ${transports}
#do
#
#	echo Synth Push
#	echo mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-compressed-NB   -o ${outroot}-synpush-${transport}-compressed-NB   -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 1 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-compressed-NB   -o ${outroot}-synpush-${transport}-compressed-NB   -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 1 -d ${synComputeParams}
#	echo mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-uncompressed-NB -o ${outroot}-synpush-${transport}-uncompressed-NB -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 1 -d ${synComputeParams}                                                                                            
#	mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-uncompressed-NB -o ${outroot}-synpush-${transport}-uncompressed-NB -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 1 -d ${synComputeParams}                                                                                            
#	echo mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-compressed      -o ${outroot}-synpush-${transport}-compressed      -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 0 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-compressed      -o ${outroot}-synpush-${transport}-compressed      -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 0 -d ${synComputeParams}
#	echo mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-uncompressed    -o ${outroot}-synpush-${transport}-uncompressed    -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 0 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Push.exe -g ${outroot}-synpush-${transport}-uncompressed    -o ${outroot}-synpush-${transport}-uncompressed    -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 0 -d ${synComputeParams}
#
#	echo Synth Full
#	echo mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-compressed-NB   -o ${outroot}-synfull-${transport}-compressed-NB   -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 1 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-compressed-NB   -o ${outroot}-synfull-${transport}-compressed-NB   -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 1 -d ${synComputeParams}
#	echo mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-uncompressed-NB -o ${outroot}-synfull-${transport}-uncompressed-NB -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 1 -d ${synComputeParams}                                                                                            
#	mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-uncompressed-NB -o ${outroot}-synfull-${transport}-uncompressed-NB -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 1 -d ${synComputeParams}                                                                                            
#	echo mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-compressed      -o ${outroot}-synfull-${transport}-compressed      -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 0 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-compressed      -o ${outroot}-synfull-${transport}-compressed      -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 1 -l 0 -d ${synComputeParams}
#	echo mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-uncompressed    -o ${outroot}-synfull-${transport}-uncompressed    -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 0 -d ${synComputeParams}
#	mpirun -np 8 ${bindir}/bin/SynData_Full.exe -g ${outroot}-synfull-${transport}-uncompressed    -o ${outroot}-synfull-${transport}-uncompressed    -n $filecount -t ${transport} -b ${buffersize} -P ${iosize} -p ${subgroup} -m ${datasize} -c 0 -l 0 -d ${synComputeParams}
#done


