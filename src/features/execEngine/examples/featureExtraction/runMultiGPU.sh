HOME_DIR=/nics/c/home/gteodor/

for i in 6;  do
	for percent in 0; do
		numGPUs=1;
		GPU_DIR=res/GPU/${percent}/${numGPUs}-gpus/$i
		mkdir -p $GPU_DIR
#		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 0 $numGPUs 1
		echo "./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000012288-0000012288-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000012288-0000012288-gray.png ${percent} 0 $numGPUs 1"


#
#		CPU_DIR=res/CPU/${percent}/$i
#		mkdir -p $CPU_DIR
#		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 0 1 > ${CPU_DIR}/res.txt
#
#		CPUGPU_DIR=res/CPUGPUFCFS/${percent}/$i
#		mkdir -p $CPUGPU_DIR
#		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 1 1 > ${CPUGPU_DIR}/res.txt
#
#
#		CPUGPU_DIR=res/CPUGPUPRIORITY/${percent}/$i
#		mkdir -p $CPUGPU_DIR
#		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 1 2 > ${CPUGPU_DIR}/res.txt
#
	done
done
