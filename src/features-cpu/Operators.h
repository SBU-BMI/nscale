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
#include "opencv2/opencv.hpp"

// Opencv GPU includes
#ifdef	USE_GPU
#include "opencv2/gpu/gpu.hpp"
#endif
using namespace std;

class Operators {
private:
	Operators(){};
	virtual ~Operators(){};

public:

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
#ifdef USE_GPU
	static unsigned int *buildHistogram256GPU(cv::gpu::GpuMat *inputImage, cv::gpu::GpuMat *inputImageMask=NULL);
#endif
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
