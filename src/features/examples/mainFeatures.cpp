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
		std::cerr<<"Parameters: imageFile maskFile <binary:1/labeled:2> outputFile\n";
		exit(-1);
	}

	char* imgFile  = argv[1];
	char* maskFile = argv[2];
	int   isBinary = atoi(argv[3]);
	char* outFile  = argv[4];

	cv::Mat inpImage;
	cv::Mat binaryMask;

	inpImage = imread(imgFile,CV_LOAD_IMAGE_COLOR);
	ImageRegionNucleiData nucleiData(0,0,inpImage.cols-1,inpImage.rows-1);
	cv::Mat_<int> labeledMask = cv::Mat_<int>::zeros(inpImage.size());
	if (isBinary) {
		binaryMask  = imread(maskFile,CV_LOAD_IMAGE_GRAYSCALE);
		labeledMask = nscale::bwlabel2(binaryMask, 8, true);
	} else {
		cv::Mat inpMask = imread(maskFile,CV_LOAD_IMAGE_COLOR);
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
		writeU24CSVFile(outFile, nucleiData);
	}

	return 0;
}
