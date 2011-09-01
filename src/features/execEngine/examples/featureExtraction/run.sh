HOME_DIR=/nics/c/home/gteodor/

IMG1=${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png
IMG1MASK=${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png

IMG2=${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000008192-0000012288-gray.png
IMG2MASK=${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000008192-0000012288-mask.png


for i in 1 2 3 4 5;  do
	for percent in 100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5 0; do

		GPU_DIR=res/GPU/${percent}/$i
		mkdir -p $GPU_DIR

		./featuresExtraction ${IMG1MASK} ${IMG1} ${IMG2MASK} ${IMG2}  ${percent} 0 1 1 > ${GPU_DIR}/res.txt

		CPU_DIR=res/CPU/${percent}/$i
		mkdir -p $CPU_DIR
		./featuresExtraction ${IMG1MASK} ${IMG1} ${IMG2MASK} ${IMG2}  ${percent} 1 0 1 > ${CPU_DIR}/res.txt

		for numCpus in 1 10 11; do

			CPUGPU_DIR=res/CPUGPUFCFS/${percent}/${numCpus}-cpus/$i
			mkdir -p $CPUGPU_DIR
			./featuresExtraction ${IMG1MASK} ${IMG1} ${IMG2MASK} ${IMG2} ${percent} ${numCpus} 1 1 > ${CPUGPU_DIR}/res.txt


			CPUGPU_DIR=res/CPUGPUPRIORITY/${percent}/${numCpus}-cpus/$i
			mkdir -p $CPUGPU_DIR
			./featuresExtraction ${IMG1MASK} ${IMG1} ${IMG2MASK} ${IMG2}  ${percent} ${numCpus} 1 2 > ${CPUGPU_DIR}/res.txt
		done
	done
done
