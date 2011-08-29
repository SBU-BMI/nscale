HOME_DIR=/nics/c/home/gteodor/

for i in 3 4 5; do
	for percent in 100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5 0; do

#	for percent in 5 0; do
		GPU_DIR=res/GPU/${percent}/$i
		mkdir -p $GPU_DIR

		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 0 1 1 > ${GPU_DIR}/res.txt

		CPU_DIR=res/CPU/${percent}/$i
		mkdir -p $CPU_DIR
		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 0 1 > ${CPU_DIR}/res.txt

		CPUGPU_DIR=res/CPUGPUFCFS/${percent}/$i
		mkdir -p $CPUGPU_DIR
		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 1 1 > ${CPUGPU_DIR}/res.txt


		CPUGPU_DIR=res/CPUGPUPRIORITY/${percent}/$i
		mkdir -p $CPUGPU_DIR
		./featuresExtraction ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ${HOME_DIR}/images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${percent} 1 1 2 > ${CPUGPU_DIR}/res.txt

	done
done
