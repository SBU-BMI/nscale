#!/bin/bash

script_dir=`dirname $0`
system="keeneland"

proc_counts="720;384;192;96"
pcs="${proc_counts//;/ }"

PREV=""
NEXT=""

for pc in ${pcs}
do
# baseline.  us PROC_COUNTS because we'd like to sort the submission order.
	for f in `ls ${script_dir}/baseline.${system}.p${pc}.*.sh`
	do
		echo "qsub -W depend=after:$PREV ${f}"
		NEXT=`qsub -W depend=after:$PREV ${f}`
		echo $NEXT
		PREV=$NEXT
	done

##adios with separate io nodes
#	for f in `ls ${script_dir}/separate.${system}.p${pc}.*.sh`
#	do
#		echo "qsub -W depend=after:$PREV ${f}"
#		NEXT=`qsub -W depend=after:$PREV ${f}`
#		echo $NEXT
#		PREV=$NEXT
#	done

##adios collocated io nodes
#	for f in `ls ${script_dir}/co-loc.${system}.p${pc}.*.sh`
#	do
#		echo "qsub -W depend=after:$PREV ${f}"
#		NEXT=`qsub -W depend=after:$PREV ${f}`
#		echo $NEXT
#		PREV=$NEXT
#	done
done


