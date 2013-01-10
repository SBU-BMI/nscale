#!/bin/bash

#grep -l -m 1 -e 'MPI NB SEND,21' -e 'MPI NB RECV,21' *
FILES=/home/tcpan/PhD/path/Data/adios/*/*.csv

for f in ${FILES}
do
	count=`grep -c -m 1 -e 'MPI NB SEND,21' -e 'MPI NB RECV,21' $f`
	if [ $count -ge 1 ]
	then 
		echo "found in $f"
		cp ${f} ${f}.mpii.bak
		perl -pi -w -e 's/MPI NB (RECV|SEND),21,/MPI NB $1,23,/g;' ${f}
		count=`grep -c -m 1 'MPI NB SEND,21\|MPI NB RECV,21' ${f}`
		echo "${f} processed.  # left: ${count}"
	fi
done

