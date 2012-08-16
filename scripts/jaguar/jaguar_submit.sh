#!/bin/bash

script_dir=`dirname $0`
system="jaguar"

PREV=`qsub ${script_dir}/warmup.sh`
echo $PREV

NEXT=""

# baseline
for f in `ls ${script_dir}/baseline.${system}.*.sh`
do
	echo "qsub -W depend=afterok:$PREV ${f}"
	NEXT=`qsub -W depend=afterok:$PREV ${f}`
	echo $NEXT
	PREV=$NEXT
done

#adios with separate io nodes
for f in `ls ${script_dir}/separate.${system}.*.sh`
do
	echo "qsub -W depend=afterok:$PREV ${f}"
	NEXT=`qsub -W depend=afterok:$PREV ${f}`
	echo $NEXT
	PREV=$NEXT
done

#adios collocated io nodes
for f in `ls ${script_dir}/co-loc.${system}.*.sh`
do
	echo "qsub -W depend=afterok:$PREV ${f}"
	NEXT=`qsub -W depend=afterok:$PREV ${f}`
	echo $NEXT
	PREV=$NEXT
done





