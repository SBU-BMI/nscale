/*
 * ObjFeatures.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#include "ObjFeatures.h"
#define SQRT2 1.41421356237
#define PI 3.14159265359

namespace nscale{


int* ObjFeatures::area(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask) {
	int* areaRes = NULL;
	if(compCount > 0){
		printf("CompCount=%d\n", compCount);
		areaRes = (int*)malloc(sizeof(int) * compCount);
		for(int i = 0; i < compCount; i++){
			//std::cout << "comp["<< i<<"] - label = "<< boundingBoxesInfo[0+i] << " minx=" <<boundingBoxesInfo[compCount+i]<< " maxx="<< boundingBoxesInfo[compCount*2+i] << " miny="<< boundingBoxesInfo[compCount*3+i]<< " maxy="<< boundingBoxesInfo[compCount*4+i] << std::endl;

			const int *labeledImgPtr;
			int area=0;
			int label=boundingBoxesInfo[i];
			int maxy = boundingBoxesInfo[compCount * 4 + i];
			int maxx = boundingBoxesInfo[compCount * 2 + i];

			for(int y = boundingBoxesInfo[compCount*3+i]; y <= maxy; y++){
				labeledImgPtr =  labeledMask.ptr<int>(y);

				for(int x = boundingBoxesInfo[compCount+i]; x <= maxx; x++){
					if(labeledImgPtr[x] == label){
						area++;
					}
				}
			}
			//std::cout << "area=" << area<<std::endl;
			areaRes[i] = area;
		}
	}

	return areaRes;

}

/*THIS FUNCTION FITS AN ELLIPSE TO THE NUCLEUS.*/
void ObjFeatures::ellipse(const int* boundingBoxesInfo,const int* areaRes, const int compCount , const cv::Mat& labeledMask, double* &majorAxis, double* &minorAxis, double* &ecc)
{
	if(compCount > 0)
	{
		majorAxis = (double *)malloc(sizeof(double) * compCount);
		minorAxis = (double *)malloc(sizeof(double) * compCount);
		ecc = (double *)malloc(sizeof(double) * compCount);
		double xbar,ybar,ssqx,ssqy,sx,sy,sxy;
		double mxx,myy,mxy;
			
		//Calculate the sums
		double frac = 1.0/12.0;
		double root = sqrt(8.0);
		const int* labeledImgPtr;
		double delta;
		int label;
		for(int i = 0; i < compCount ; i++)
		{
			ssqx = 0.0;
			ssqy = 0.0;
			sx = 0.0;
			sy = 0.0;
			sxy = 0.0;
			label = boundingBoxesInfo[i];
			
			int minX = boundingBoxesInfo[compCount + i];
			int maxX = boundingBoxesInfo[2 * compCount + i];
			float midX = (float)(minX+maxX)/2.0;
			int minY = boundingBoxesInfo[3 * compCount + i];
			int maxY = boundingBoxesInfo[4 * compCount + i];
			float midY = (float)(minY+maxY)/2.0;
	
			
			
			//Walk through the tile
			for(int y = minY ; y <= maxY; y++)
			{
				float cy = (float)y - midY;
				labeledImgPtr = labeledMask.ptr<int>(y);
				for(int x = minX ; x<= maxX ; x++)
				{
					float cx = (float)x - midX;
					if (labeledImgPtr[x] == label) {
						sx += cx;
						sy += cy;
						sxy += cx*cy;
						ssqx += cx*cx;
						ssqy += cy*cy;
					}
				}
			}
			
			//Calculate mxx,myy,mxy,xbar,ybar
			xbar = sx/areaRes[i];
			ybar = sy/areaRes[i];
			
			mxx = ((ssqx + areaRes[i] * xbar * xbar - 2.0 * xbar * sx)/areaRes[i]) + frac;
		  	myy = ((ssqy + areaRes[i] * ybar * ybar - 2.0 * ybar * sy)/areaRes[i]) + frac;
			mxy = (sxy - ybar * sx - xbar * sy + xbar * ybar * areaRes[i])/areaRes[i];
	
			//Calculate the major axis, minor axis and eccentricity
			delta = sqrt((mxx-myy)*(mxx-myy) + 4.0 * mxy * mxy); //discriminant = sqrt(b*b-4*a*c)
			majorAxis[i] = root*sqrt(mxx+myy+delta);
			minorAxis[i] = root*sqrt(mxx+myy-delta);
			ecc[i] = (2.0 * sqrt((majorAxis[i] * majorAxis[i] - minorAxis[i] * minorAxis[i])/4.0))/majorAxis[i];
		}
	}
	return;
}

double *ObjFeatures::extent_ratio(const int* boundingBoxesInfo, const int compCount, const int* areaRes)
{
	if(compCount > 0)
	{
		//Declare an extent_ratio array of size compCount for all the components in the slide/tile
		double *extent_ratio;
		extent_ratio = (double *)malloc(compCount * sizeof(double));
		int width;
		int height;
		for(int i = 0; i <compCount ; i++)
		{
			width = boundingBoxesInfo[compCount * 2 + i] - boundingBoxesInfo[compCount + i];
			height = boundingBoxesInfo[compCount * 4 + i] - boundingBoxesInfo[compCount * 3 + i];
			extent_ratio[i] = (double)areaRes[i] / (double)(width * height);
		}
		
		//Return the extent ratio vector
		return extent_ratio;
	}
	//If the component count is 0, then return a null pointer
	return NULL;
}

/*THIS FUNCTION CALCULATES THE PERIMETER OF EVERY OBJECT IN THE IMAGE AND STORES IT IN AN ARRAY CALLED perimeterRes*/
/*perimeterRes[i] = perimeter of the object with label boundingBoxInfo[i]*/
//This function calculates the perimeter of a component using a modified version of the marching squares algorithm
double *ObjFeatures::perimeter(const int* boundingBoxesInfo, const int compCount,const cv::Mat& labeledMask)
{
	if(compCount > 0)
	{
		double *perimeter;
		perimeter = (double *)malloc(compCount * sizeof(double));
		double *lookup;
		lookup = (double *)calloc(16,sizeof(double));
		lookup[8] = 0.70710678118;
		lookup[4] = 0.70710678118;
		lookup[2] = 0.70710678118;
		lookup[1] = 0.70710678118;
		lookup[3] = 1.0;
		lookup[6] = 1.0;
		lookup[9] = 1.0;
		lookup[12] = 1.0;
		lookup[7] = 0.70710678118;
		lookup[11] = 0.70710678118;
		lookup[13] = 0.70710678118;
		lookup[14] = 0.70710678118;
		lookup[10] = SQRT2;
		lookup[5] = SQRT2;
		const int* labeledImgPtr_ybot;
		const int* labeledImgPtr_ytop;
                int label;
	
		//Walk through each bounding box
		for(int i = 0 ; i < compCount ; i++)
		{
			label = boundingBoxesInfo[i];
			perimeter[i] = 0.0;
			int xmin = boundingBoxesInfo[compCount + i];
			int xmax = boundingBoxesInfo[compCount * 2 + i];
			int ymin = boundingBoxesInfo[compCount * 3 + i];
			int ymax = boundingBoxesInfo[compCount * 4 + i];
			uint8_t mask;//This is the mask that will hold the prefix

			//Traverse the centre of the image
			for(int y = ymin; y < ymax; y++)
			{
				labeledImgPtr_ybot = labeledMask.ptr<int>(y);
				labeledImgPtr_ytop = labeledMask.ptr<int>(y+1);
				
				for(int x = xmin; x < xmax; x++)
				{
					mask = 0;
					
					//First point : Lower left corner (0,0)
					mask = (labeledImgPtr_ybot[x] == label);
					//Second point : Lower right corner (1,0)
					mask = (mask<<1) | (labeledImgPtr_ybot[x+1] == label);
					//Third point : Upper left corner (1,1)
					mask = (mask<<1) | (labeledImgPtr_ytop[x+1] == label);
					//Fourth point : Upper right corner (0,1)
					mask = (mask<<1) | (labeledImgPtr_ytop[x] == label);
					
					perimeter[i] = perimeter[i] + lookup[mask];
				} 
			}
	
			//Traverse the top and bottom edges of the image
			labeledImgPtr_ybot = labeledMask.ptr<int>(ymin);
			labeledImgPtr_ytop = labeledMask.ptr<int>(ymax);
			for(int x = xmin; x < xmax; x++)
			{
				mask = 0;
				//Top row : Read-leftshift-read-leftshiftby2
				mask = (labeledImgPtr_ytop[x] == label);
				mask = (mask<<1) | (labeledImgPtr_ytop[x+1] == label);
				mask = (mask<<2);
				
				
				perimeter[i] = perimeter[i] + lookup[mask];
	
				mask = 0;
				//Bottom row : Leftshiftby2-read-leftshift-read
				mask = (labeledImgPtr_ybot[x+1] == label);
				mask = (mask<<1) | (labeledImgPtr_ybot[x] == label);
	
				perimeter[i] = perimeter[i] + lookup[mask];
			} 
	
			//Traverse the left and right edges of the images
			for(int y = ymin ; y < ymax ; y++)
			{
				labeledImgPtr_ybot = labeledMask.ptr<int>(y);
				labeledImgPtr_ytop = labeledMask.ptr<int>(y+1);
				
				mask = 0;
				//Left edge : leftshift-read-leftshift-read-leftshift
				mask = (labeledImgPtr_ybot[xmin] == label);
				mask = (mask<<1) | (labeledImgPtr_ytop[xmin] == label);
				mask = (mask<<1);
				
				perimeter[i] = perimeter[i] + lookup[mask];
	
				mask = 0;
				//Right edge : read-leftshiftby3-read
				mask = (labeledImgPtr_ybot[xmax] == label);
				mask = (mask << 2);
				mask = (mask << 1) | (labeledImgPtr_ytop[xmax] == label);
		
				perimeter[i] = perimeter[i] + lookup[mask];
			}
	
			//Bottom left corner of the image : leftshiftby3-read-leftshift
			mask = 0;
			mask = (labeledMask.ptr<int>(ymin)[xmin] == label);
			mask = (mask<<1);
			perimeter[i] = perimeter[i] + lookup[mask];
	
			//Bottom right corner of the image : leftshiftby3-read
			mask = 0;
			mask = (labeledMask.ptr<int>(ymin)[xmax] == label);
			perimeter[i] = perimeter[i] + lookup[mask];
	
			//Top right corner of the image : read-leftshiftby3
			mask  = 0;
			mask = (labeledMask.ptr<int>(ymax)[xmax] == label);
			mask = (mask<<3);
			perimeter[i] = perimeter[i] + lookup[mask];
	
			//Top left corner of the image : leftshift-read-leftshiftby2
			mask = 0;
			mask = (labeledMask.ptr<int>(ymax)[xmin] == label);
			mask = (mask<<2);
			perimeter[i] = perimeter[i] + lookup[mask];
			
		} 
		return perimeter;	
	}	
	return NULL;
}
	
//This function calculates the circularity of a particular component
double* ObjFeatures::circularity(const int compCount, const int* areaRes, const double* perimeter)
{
	if(compCount > 0)
	{
		double* circ = (double *)malloc(compCount * sizeof(double));
		for(int i = 0 ; i < compCount ; i++)
		{
			circ[i] = (4.0 * PI * (double)areaRes[i])/(perimeter[i] * perimeter[i]);	
		}
		
		return circ;
	}
	return NULL;
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

