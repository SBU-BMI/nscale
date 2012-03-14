/*
 * Operators.h
 *
 *  Created on: Jul 21, 2011
 *      Author: george
 */

#ifndef OPERATORS_H_
#define OPERATORS_H_

#include <math.h>
#include <cmath>
#include <iostream>

// Includes used by opencv
#include "highgui.h"
#include "cv.h"
#include "cxcore.h"

// Opencv GPU includes
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;

class Operators {
private:
	Operators(){};
	virtual ~Operators(){};


public:
	static void gradient(cv::Mat& inputImageMat, cv::Mat& gradientMat);

	//Co-occurence related operators
	static float calcMxFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float calcMyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float inertiaFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float energyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float entropyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float homogeneityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float maximumProbabilityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static float clusterShadeFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k=1);
	static float clusterProminenceFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k=1);

	//Histogram related operators
	static unsigned int *buildHistogram256CPU(IplImage *inputImage, IplImage *inputImageMask=NULL);

	static unsigned int *buildHistogram256GPU(cv::gpu::GpuMat *inputImage, cv::gpu::GpuMat *inputImageMask=NULL);
	static double calcMeanFromHistogram(int *hist, int numBins);
	static double calcStdFromHistogram( int *hist,  int numBins);
	static int calcMedianFromHistogram( int *hist,  int numBins);
	static int calcMinFromHistogram( int *hist,  int numBins);
	static int calcMaxFromHistogram( int *hist,  int numBins);
	static int calcFirstQuartileFromHistogram( int *hist,  int numBins);
	static int calcSecondQuartileFromHistogram( int *hist,  int numBins);
	static int calcThirdQuartileFromHistogram( int *hist,  int numBins);
	static int calcNonZeroFromHistogram( int *hist,  int numBins);

	static float calcEntropyFromHistogram(int *hist,  int numBins);
	static float calcEnergyFromHistogram(int *hist,  int numBins);
	static float calcSkewnessFromHistogram(int *hist,  int numBins);
	static float calcKurtosisFromHistogram(int *hist,  int numBins);



	static int calcNumElementsFromHistogram( int *hist,  int numBins);

	//  operators per obj
	static int* buildHistogram256CPU( const cv::Mat& labeledMask, const cv::Mat& grayImage, int minx,  int maxx,  int miny,  int maxy,  int label );


};

#endif /* OPERATORS_H_ */
