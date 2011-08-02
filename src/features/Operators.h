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

	//Co-occurence related operators
	static double calcMxFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double calcMyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double inertiaFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double energyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double entropyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double homogeneityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double maximumProbabilityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount);
	static double clusterShadeFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k=1);
	static double clusterProminenceFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k=1);
//	static double correlationFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k=1);

	//Histogram related operators
	static unsigned int *buildHistogram256CPU(IplImage *inputImage, IplImage *inputImageMask=NULL);
	static unsigned int *buildHistogram256GPU(cv::gpu::GpuMat *inputImage, cv::gpu::GpuMat *inputImageMaks=NULL);
	static double calcMeanFromHistogram(unsigned int *hist, unsigned int numBins);
	static double calcStdFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcMedianFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcMinFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcMaxFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcFirstQuartileFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcSecondQuartileFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcThirdQuartileFromHistogram(unsigned int *hist, unsigned int numBins);
	static int calcNumElementsFromHistogram(unsigned int *hist, unsigned int numBins);

	//  operators


};

#endif /* OPERATORS_H_ */
