#!/bin/bash

#grep -l -m 1 -e 'toRGB,0' *
#for f in *.csv; do echo "${f}: `grep -o -e 'toRGB,0' ${f} | wc -l`"; done
FILES=/home/tcpan/PhD/path/Data/adios/*/*.csv

# convert from the following:
#toRGB,0,39761956,39796521,1,background,0,39796556,39835035,1,RBC,0,39835205,40155364,1,GrayNU,0,40155379,41959561,1,NuMask,0,41959576,42997635,1,removeRBC,0,42997663,44000955,1,separateNuclei,0,44000967,48526324,1,finalCleanup,0,48526339,50074996,1
# to computeFull,0,xxxx,yyyy,1
#toRGB,0,39761956,39796521,1,background,0,39796556,39835035,1,RBC,0,39835205,40155364,1,GrayNU,0,40155379,41959561,1,NuMask,0,41959576,42997635,1,removeRBC,0,42997663,44000955,1
# to computeNoNU,0,xxxx,yyyy,1
#toRGB,0,39761956,39796521,1,background,0,39796556,39835035,1
# to computeNoFG,0,xxxx,yyyy,1


for f in ${FILES}
do
	count=`grep -c -m 1 -e 'toRGB,0' $f`
	if [ $count -ge 1 ]
	then 
		echo "found in $f"
		cp ${f} ${f}.compStages.bak
		perl -pi -w -e 's/toRGB,0,(\d+),\d+,1,background,0,\d+,\d+,1,RBC,0,\d+,\d+,1,GrayNU,0,\d+,\d+,1,NuMask,0,\d+,\d+,1,removeRBC,0,\d+,\d+,1,separateNuclei,0,\d+,\d+,1,finalCleanup,0,\d+,(\d+),1/computeFull,0,$1,$2,1/g;' ${f}
		count=`grep -c -m 1 ',finalCleanup,' ${f}`
		echo "${f} processed.  # left: ${count}"
		perl -pi -w -e 's/toRGB,0,(\d+),\d+,1,background,0,\d+,\d+,1,RBC,0,\d+,\d+,1,GrayNU,0,\d+,\d+,1,NuMask,0,\d+,\d+,1,removeRBC,0,\d+,(\d+),1/computeNoNU,0,$1,$2,1/g;' ${f}
		count=`grep -c -m 1 ',removeRBC,' ${f}`
		echo "${f} processed.  # left: ${count}"
		perl -pi -w -e 's/toRGB,0,(\d+),\d+,1,background,0,\d+,(\d+),1/computeNoFG,0,$1,$2,1/g;' ${f}
		count=`grep -c -m 1 ',background,' ${f}`
		echo "${f} processed.  # left: ${count}"
	fi
done

