#include <cstdio>
#include <iostream>
#include <string>
#include <omp.h>

// openCV
#include <opencv2/opencv.hpp>

// Feature computation
#include "ImageRegionNucleiData.h" 
#include "ImageRegionNucleiDataIO.h"

int main(int argc, char **argv) 
{
	if (argc < 4) {
		std::cerr<<"Parameters: imageFile maskFile <1 (for binary mask) or 0 (for labeled mask image)> outputFile\n";
		exit(-1);
	}

	char* imgFile  = argv[1];
	char* maskFile = argv[2];
	int   isBinary = atoi(argv[3]);
	char* outFile  = argv[4];


	cv::Mat inpImage = imread(imgFile,CV_LOAD_IMAGE_COLOR);
	cv::Mat inpMask = imread(maskFile,-1);

	ImageRegionNucleiData nucleiData(0,0,inpImage.cols-1,inpImage.rows-1);
	cv::Mat_<int> labeledMask = cv::Mat_<int>::zeros(inpImage.size());

	if (isBinary) {
		printf("BinaryMask --> Mask depth (# channels): %d\n",inpMask.channels());
		cv::Mat binaryMask = Mat::zeros(inpMask.size(),CV_8U);
		if (inpMask.channels()==2) {
			for(int y=0;y<inpMask.rows;y++) {
				for(int x=0;x<inpMask.cols;x++) {
					binaryMask.at<unsigned char>(Point(x,y)) = (unsigned char) inpMask.at<short>(Point(x,y));
				}
			}
		} else if (inpMask.channels()==1) {
			binaryMask = inpMask;
		} else { 
			fprintf(stderr, "Binary Mask file has more than 2 channels.\n");
			return 1;
		}
		labeledMask = nscale::bwlabel2(binaryMask, 8, true);
	} else {
		printf("Labeled Mask --> Mask depth (# channels): %d\n",inpMask.channels());
		for(int y=0;y<inpMask.rows;y++) {
			for(int x=0;x<inpMask.cols;x++) {
				Vec3b color = inpMask.at<Vec3b>(Point(x,y));
			    labeledMask.at<int>(Point(x,y)) = (int)(color[0]*256*256+color[1]*256+color[2]);	
			}
		}
	}
	nucleiData.extractBoundingBoxesFromLabeledMask(labeledMask);
		
	std::cout << "COMP COUNT: " << nucleiData.getNumberOfNuclei() << std::endl;
		
	if (nucleiData.getNumberOfNuclei()>0) {
		nucleiData.extractPolygonsFromLabeledMask(labeledMask); 
		nucleiData.extractCytoplasmRegions(labeledMask);
		nucleiData.computeShapeFeatures(labeledMask);
		nucleiData.computeRedBlueChannelTextureFeatures(inpImage,labeledMask);
		writeU24CSVFileFromVector(outFile, nucleiData);
	}

	return 0;
}
