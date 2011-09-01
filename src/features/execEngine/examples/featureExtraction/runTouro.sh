for i in 1 2 3; do
	for perc in 0 5 10 15 20 25 30; do

		GPU_DIR=res/GPU/${perc}/$i
		mkdir $GPU_DIR
		./featuresExtraction ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${perc} 0 1 1 > ${GPU_DIR}/res.txt

#		CPU_DIR=res/CPU/$i
#		mkdir $CPU_DIR
#		./featuresExtraction ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${perc} 0 1 1 > ${CPU_DIR} > res.txt

		CPUGPU_DIR=res/CPUGPUFCFS/${perc}/$i
		mkdir -p $CPUGPU_DIR
		./featuresExtraction ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${perc} 1 1 1  > ${CPUGPU_DIR}/res.txt

		CPUGPU_DIR=res/CPUGPUPRIORITY/${perc}/$i
		mkdir -p $CPUGPU_DIR
		./featuresExtraction ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000004096-gray.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-mask.png ../../../images/TCGA-06-0216-01B-01-HF1057-3.ndpi-0000032768-0000008192-gray.png ${perc} 1 1 1 > ${CPUGPU_DIR}/res.txt

	done
done
