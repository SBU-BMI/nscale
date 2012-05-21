/*
 * ObjFeatures.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#include "ObjFeatures.h"

namespace nscale{


int* ObjFeatures::area(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask) {
	int* areaRes = NULL;
	if(compCount > 0){
		printf("CompCount=%d\n", compCount);
		areaRes = (int*)malloc(sizeof(int) * compCount);
		for(int i = 0; i < compCount; i++){
			std::cout << "comp["<< i<<"] - label = "<< boundingBoxesInfo[0+i] << " minx=" <<boundingBoxesInfo[compCount+i]<< " maxx="<< boundingBoxesInfo[compCount*2+i] << " miny="<< boundingBoxesInfo[compCount*3+i]<< " maxy="<< boundingBoxesInfo[compCount*4+i] << std::endl;

			const int *labeledImgPtr;
			int area=0;
			int label=boundingBoxesInfo[i];
			for(int y = boundingBoxesInfo[compCount*3+i]; y <= boundingBoxesInfo[compCount*4+i]; y++){
				labeledImgPtr =  labeledMask.ptr<int>(y);

				for(int x = boundingBoxesInfo[compCount+i]; x <= boundingBoxesInfo[compCount*2+i]; x++){
					if(labeledImgPtr[x] == label){
						area++;
					}
				}
			}
			std::cout << "area=" << area<<std::endl;
		}
	}

	return areaRes;

}
float* ObjFeatures::cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage) {
	float* intensityFeatures = NULL;
//	printf("intensityFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		intensityFeatures = (float*)malloc(sizeof(float) * compCount * N_INTENSITY_FEATURES);
		for(int i = 0; i < compCount; i++){
			int label = 255;
			int dataOffset = boundingBoxesInfo[i*5];
			int minx = boundingBoxesInfo[i*5+1];
			int miny = boundingBoxesInfo[i*5+2];
			int width = boundingBoxesInfo[i*5+3];
			int height = boundingBoxesInfo[i*5+4];
			int maxx = minx+width-1;
			int maxy = miny+height-1;

			// Points to address where cytoplasm masks supposed to be stored: root address + dataOffset
			char *dataAddress = ((char*)(boundingBoxesInfo))+dataOffset;

			// Create a Mat header point to the data we allocate
			cv::Mat objMask(height, width, CV_8UC1, dataAddress );

			int* compHist = Operators::buildHistogram256CPUObjMask(objMask, grayImage, minx, maxx, miny, maxy, label, i);
//			if(i == 0){
//				printf("Label = %d\n", label);
//			}
//			if(i == 0 ){
//				for(int j = 0; j < 256; j++){
//					printf("hist[%d]=%d\n", j, compHist[j]);
//				}
//			}
			intensityFeatures[0 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);
			intensityFeatures[1 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMaxFromHistogram(compHist, 256);
			intensityFeatures[2 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMinFromHistogram(compHist, 256);
			intensityFeatures[3 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcStdFromHistogram(compHist, 256);
			intensityFeatures[4 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcEntropyFromHistogram(compHist, 256);
			intensityFeatures[5 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcEnergyFromHistogram(compHist, 256);
			intensityFeatures[6 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcSkewnessFromHistogram(compHist, 256);
			intensityFeatures[7 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcKurtosisFromHistogram(compHist, 256);

			free(compHist);
		}

	}
	return intensityFeatures;
}

float* ObjFeatures::intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage) {
	float* intensityFeatures = NULL;
//	printf("intensityFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		intensityFeatures = (float*)malloc(sizeof(float) * compCount * N_INTENSITY_FEATURES);
		for(int i = 0; i < compCount; i++){
			int label = boundingBoxesInfo[i];
			int minx = boundingBoxesInfo[compCount+i];
			int maxx = boundingBoxesInfo[compCount*2+i];
			int miny = boundingBoxesInfo[compCount*3+i];
			int maxy = boundingBoxesInfo[compCount*4+i];

			int* compHist = Operators::buildHistogram256CPU(labeledMask, grayImage, minx, maxx, miny, maxy, label);

//			if(i == 0 || i == 1){
//				for(int j = 0; j < 256; j++){
//					printf("hist[%d]=%d\n", j, compHist[j]);
//				}
//			}
			intensityFeatures[0 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);
			intensityFeatures[1 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMaxFromHistogram(compHist, 256);
			intensityFeatures[2 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcMinFromHistogram(compHist, 256);
			intensityFeatures[3 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcStdFromHistogram(compHist, 256);
			intensityFeatures[4 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcEntropyFromHistogram(compHist, 256);
			intensityFeatures[5 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcEnergyFromHistogram(compHist, 256);
			intensityFeatures[6 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcSkewnessFromHistogram(compHist, 256);
			intensityFeatures[7 + ObjFeatures::N_INTENSITY_FEATURES * i] = Operators::calcKurtosisFromHistogram(compHist, 256);

			free(compHist);
		}

	}
	return intensityFeatures;
}

float* ObjFeatures::gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage) {
	float* gradientFeatures = NULL;
//	printf("gradientFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		gradientFeatures = (float*)malloc(sizeof(float) * compCount * N_GRADIENT_FEATURES);

		cv::Mat gradientMat;
		Operators::gradient((cv::Mat&)grayImage, gradientMat);

		for(int i = 0; i < compCount; i++){
			int label = boundingBoxesInfo[i];
			int minx = boundingBoxesInfo[compCount+i];
			int maxx = boundingBoxesInfo[compCount*2+i];
			int miny = boundingBoxesInfo[compCount*3+i];
			int maxy = boundingBoxesInfo[compCount*4+i];



			int* compHist = Operators::buildHistogram256CPU(labeledMask, gradientMat, minx, maxx, miny, maxy, label);


//			if(i == 0 || i == 1){
//				for(int j = 0; j < 256; j++){
//					printf("hist[%d]=%d\n", j, compHist[j]);
//				}
//			}

			gradientFeatures[0 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);
			gradientFeatures[1 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcStdFromHistogram(compHist, 256);
			gradientFeatures[2 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcEntropyFromHistogram(compHist, 256);
			gradientFeatures[3 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcEnergyFromHistogram(compHist, 256);
			gradientFeatures[4 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcSkewnessFromHistogram(compHist, 256);
			gradientFeatures[5 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcKurtosisFromHistogram(compHist, 256);

			free(compHist);
		}
		gradientMat.release();

	}
	return gradientFeatures;
}


float* ObjFeatures::cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage) {
	float* gradientFeatures = NULL;
//	printf("gradientFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		gradientFeatures = (float*)malloc(sizeof(float) * compCount * N_GRADIENT_FEATURES);

		cv::Mat gradientMat;
		Operators::gradient((cv::Mat&)grayImage, gradientMat);

		for(int i = 0; i < compCount; i++){
			int label = 255;
			int dataOffset = boundingBoxesInfo[i*5];
			int minx = boundingBoxesInfo[i*5+1];
			int miny = boundingBoxesInfo[i*5+2];
			int width = boundingBoxesInfo[i*5+3];
			int height = boundingBoxesInfo[i*5+4];
			int maxx = minx+width-1;
			int maxy = miny+height-1;

			// Points to address where cytoplasm masks supposed to be stored: root address + dataOffset
			char *dataAddress = ((char*)(boundingBoxesInfo))+dataOffset;

			// Create a Mat header point to the data we allocate
			cv::Mat objMask(height, width, CV_8UC1, dataAddress );

			int* compHist = Operators::buildHistogram256CPUObjMask(objMask, gradientMat, minx, maxx, miny, maxy, label, i);


//			if(i == 0 || i == 1){
//				for(int j = 0; j < 256; j++){
//					printf("hist[%d]=%d\n", j, compHist[j]);
//				}
//			}

			gradientFeatures[0 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);
			gradientFeatures[1 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcStdFromHistogram(compHist, 256);
			gradientFeatures[2 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcEntropyFromHistogram(compHist, 256);
			gradientFeatures[3 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcEnergyFromHistogram(compHist, 256);
			gradientFeatures[4 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcSkewnessFromHistogram(compHist, 256);
			gradientFeatures[5 + ObjFeatures::N_GRADIENT_FEATURES * i] = Operators::calcKurtosisFromHistogram(compHist, 256);

			free(compHist);
		}
		gradientMat.release();

	}
	return gradientFeatures;
}




float* ObjFeatures::cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage) {
	float* cannyFeatures = NULL;
//	printf("gradientFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		cannyFeatures = (float*)malloc(sizeof(float) * compCount * N_CANNY_FEATURES);

		cv::Mat cannyRes(grayImage.size(), grayImage.type());
		cv::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

		for(int i = 0; i < compCount; i++){
			int label = boundingBoxesInfo[i];
			int minx = boundingBoxesInfo[compCount+i];
			int maxx = boundingBoxesInfo[compCount*2+i];
			int miny = boundingBoxesInfo[compCount*3+i];
			int maxy = boundingBoxesInfo[compCount*4+i];

			int* compHist = Operators::buildHistogram256CPU(labeledMask, cannyRes, minx, maxx, miny, maxy, label);

			cannyFeatures[0 + ObjFeatures::N_CANNY_FEATURES * i] = Operators::calcNonZeroFromHistogram(compHist, 256);
			cannyFeatures[1 + ObjFeatures::N_CANNY_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);

			free(compHist);
		}
		cannyRes.release();

	}
	return cannyFeatures;
}

float* ObjFeatures::cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage) {
	float* cannyFeatures = NULL;
//	printf("gradientFeatures. compcount=%d\n", compCount);
	if(compCount > 0){
		cannyFeatures = (float*)malloc(sizeof(float) * compCount * N_CANNY_FEATURES);

		cv::Mat cannyRes(grayImage.size(), grayImage.type());
		cv::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

		for(int i = 0; i < compCount; i++){
			int label = 255;
			int dataOffset = boundingBoxesInfo[i*5];
			int minx = boundingBoxesInfo[i*5+1];
			int miny = boundingBoxesInfo[i*5+2];
			int width = boundingBoxesInfo[i*5+3];
			int height = boundingBoxesInfo[i*5+4];
			int maxx = minx+width-1;
			int maxy = miny+height-1;

			// Points to address where cytoplasm masks supposed to be stored: root address + dataOffset
			char *dataAddress = ((char*)(boundingBoxesInfo))+dataOffset;

			// Create a Mat header point to the data we allocate
			cv::Mat objMask(height, width, CV_8UC1, dataAddress );

			int* compHist = Operators::buildHistogram256CPUObjMask(objMask, cannyRes, minx, maxx, miny, maxy, label, i);

			cannyFeatures[0 + ObjFeatures::N_CANNY_FEATURES * i] = Operators::calcNonZeroFromHistogram(compHist, 256);
			cannyFeatures[1 + ObjFeatures::N_CANNY_FEATURES * i] = Operators::calcMeanFromHistogram(compHist, 256);

			free(compHist);
		}
		cannyRes.release();

	}
	return cannyFeatures;
}

}

