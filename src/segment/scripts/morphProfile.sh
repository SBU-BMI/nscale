#!/bin/sh

for nPass in `seq 1 2 30`; 
do
	for nImages in `seq 1 28`;
	do
		./imreconTest.exe $nImages $nPass >> profile-4.txt
	done
done
