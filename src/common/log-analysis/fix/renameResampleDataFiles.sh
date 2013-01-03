#!/bin/bash

for f in /home/tcpan/PhD/path/Data/adios/*/*.resampledata_by_event.mat
do
	mv -v $f ${f%.resampledata_by_event.mat}.events_resample.mat
done

