#!/bin/bash

#grep -rl 'IO POSIX Write,[0-9][0-9]*,[0-9][0-9]*,[0-9][0-9]*,,' *
FILES="cci-gpu-clus-async-blocking-nas/test-128.4.32-1.8-1-na-POSIX.csv
cci-gpuclus-b-compressed/test-128.4.32-1.8-1-na-POSIX.csv
cci-gpuclus-blocking/test-128.4.32-1.8-1-na-POSIX.csv
jaguar-hpdc2012-weak1/TCGA.separate.jaguar.p10240.f71680.na-POSIX.b4.io300-1.is1.csv
jaguar-tcga-transports/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is2560-1.csv
jaguar-tcga-transports/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is16-1.csv
jaguar-tcga-transports/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is1-1.csv
jaguar-tcga-transports-sep_osts/na-POSIX.csv
jaguar-tcga-transports-sep_osts/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is2560-1.csv
jaguar-tcga-transports-sep_osts/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is16-1.csv
jaguar-tcga-transports-sep_osts/TCGA.separate.jaguar.p10240.f100000.na-POSIX.b4.io2560-16.is1-1.csv
keeneland-hpdc2012-1/TCGA.separate.keeneland.n32.f9600.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-1/TCGA.separate.keeneland.n60.f18000.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-1/TCGA.separate.keeneland.n32.f9600.na-POSIX.b4.io60-1.is1.ost.csv
keeneland-hpdc2012-1/TCGA.separate.keeneland.n60.f18000.na-POSIX.b4.io60-1.is1.ost.csv
keeneland-hpdc2012-2/TCGA.separate.keeneland.n32.f9600.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-2/TCGA.separate.keeneland.n32.f9600.na-POSIX.b4.io60-1.is1.ost.csv
keeneland-hpdc2012-2/TCGA.separate.keeneland.n100.f30000.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-3/TCGA.separate.keeneland.n60.f18000.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-3/TCGA.separate.keeneland.n60.f18000.na-POSIX.b4.io60-1.is1.ost.csv
keeneland-hpdc2012-4/TCGA.separate.keeneland.n32.f9600.na-POSIX.b4.io60-1.is1.csv
keeneland-hpdc2012-4/TCGA.separate.keeneland.n60.f18000.na-POSIX.b4.io60-1.is1.csv
yellowstone-async/posix_raw-8.4.4-1.4-4.csv
yellowstone-async/astro-5.4.2-1.2-1-na-POSIX.csv
yellowstone-async-buf/astro-5.4.2-1.2-1-na-POSIX.csv
yellowstone-async-buf2/astro-5.4.2-1.2-1-na-POSIX.csv
yellowstone-B-compressed/astro-8.4.1-1.1-1-na-POSIX.csv
yellowstone-B-compressed/astro-8.4.2-1.2-1-na-POSIX.csv
yellowstone-B-compressed/astro-8.4.2-1.1-1-na-POSIX.csv
yellowstone-NB-buf/astro-3.4.1-1.1-1-na-POSIX.csv
yellowstone-NB-buf4/astro-8.4.3-1.1-1-na-POSIX.csv
yellowstone-NB-buf4/astro-8.4.1-1.1-1-na-POSIX.csv
yellowstone-NB-buf4/astro-8.4.3-1.3-1-na-POSIX.csv
yellowstone-NB-buf5/astro-8.4.3-1.3-1-na-POSIX.csv"

for f in ${FILES}
do
	cp ${f} ${f}.bak
	perl -pi -w -e 's/(IO POSIX Write,\d+,\d+,\d+),,/$1,67108864,/g;' ${f}	
	sed -r -n '/IO POSIX Write,[0-9]+,[0-9]+,[0-9]+,,/p' ${f}
done

