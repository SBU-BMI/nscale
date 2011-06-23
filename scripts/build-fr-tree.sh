#!/bin/bash
#
# Generate a script suitable for submission to sun grid engine
# which will generate the filter response for a collection of
# tiff regions
#
# Parameters:
#
# $1 = path to directory containing region collections/whole slides
# $2 = tif extension; if not present .tiff assumed
#

if [ $# -eq 0 ]
then
  echo "Usage: `basename $0` options (-pe)"
  exit 85
fi

while getopts ":p:e:" Option
do
  case $Option in
      p ) INPATH=${OPTARG} ;;
      e ) FEXT=${OPTARG} ;;
      * ) echo "Unrecognized option ${OPTARG}." ;;
      esac
done

for slide in `ls -1 ${INPATH-./}`; do
    if [ -d $INPATH/$slide ]; then
	mkdir -p $slide
	cd $slide
	rm -f tile-list
	ls -1 $INPATH/$slide/*.$FEXT > tile-list
	C=`wc -l < tile-list`
	sed -e "s/\$COUNT/$C/g" -e "s/\$E/$FEXT/g" > gen-fr-sge.sh <<'EOF'
#$ -cwd
#$ -S /bin/bash

#$ -t 1-$COUNT

INFILE=`awk "NR==$SGE_TASK_ID" tile-list`

/opt/local/cci/bin/fr -f $INFILE -o ./ -s 10 -e $E
EOF
	cd ..
    fi
done
