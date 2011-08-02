/*
 * RegionalMorphologyAnalysis.h
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#ifndef REGIONALMORPHOLOGYANALYSIS_H_
#define REGIONALMORPHOLOGYANALYSIS_H_

#include "Blob.h"
#include "Contour.h"
#include "Operators.h"
#include <iomanip>
#include <sys/time.h>

// Includes used by opencv
#include "highgui.h"
#include "cv.h"
#include "cxcore.h"

// Includes to use opencv2/GPU
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>

using namespace std;
using namespace cv;

using namespace cv;

//! Number of angles in which the coocurrence matrices are calculated
#define 	NUM_ANGLES	4

//! Constants used to calculate coocurrence matrices
#define		ANGLE_0		0
#define		ANGLE_45	1
#define 	ANGLE_90	2
#define 	ANGLE_135	3

//! Defining what processor should be used when invoking the functions
#define		CPU			1
#define		GPU			2

class RegionalMorphologyAnalysis {
private:

	//! Vector holding the blob identified in the input image
	vector<Blob *> internal_blobs;

	//! Pointer to a in-memory copy of the input image
	IplImage *originalImage;

	//! Pointer to a in-memory copy of the input image mask
	IplImage *originalImageMask;

	//! Pointer to a copy of the input image in the GPU memory.
	// If not NULL its pointer refers to the copy in the GPU main memory
	cv::gpu::GpuMat *originalImageGPU;

	//! Pointer to a copy of the input image mask in the GPU memory.
	// If not NULL its pointer refers to the copy in the GPU main memory
	cv::gpu::GpuMat *originalImageMaskGPU;

	//! Matrices holding coocurrence matrix computed from the input image (8x8) by default.
	// The matrix associated to each angle is stored in [ANGLE_] index.
	unsigned int **coocMatrix;

	//! Vector containing the number of element in each coocurrence matrix above
	unsigned int *coocMatrixCount;

	//! Size of the coocurrence matrix in each dimension. It is 8 by default.
	unsigned int coocSize;

	//! Vector containing the intensity histogram of the input image (256 bins), whether not NULL.
	unsigned int *intensity_hist;

	//! Vector containing the gradient magnitude histogram of the input image (256 bins), whether not NULL.
	unsigned int *gradient_hist;

	/*!
	 * Function responsible for identifying contour into an image, and instantiate the respective blobs
	 */
	void initializeContours();

	/*!
	 * Simply return a pointer to the image mask
	 */
	IplImage *getMask();

	/*!
	 * Constructor is declared private to avoid the class from be initialized
	 * without performing the appropriate initialization from the input image.
	 */
	RegionalMorphologyAnalysis();

	/*!
	 * Function that calculates the image morphological gradient, and build an histogram from the resulting image.
	 */
	unsigned int *calcGradientHistogram(bool useMask, int procType, bool reuseItermediaryResults, int gpuId);
public:
	RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName);
	virtual ~RegionalMorphologyAnalysis();

	/*!
	 * Computes features similar to matlab regionpros, from each blob found in input image.
	 */
	void doRegionProps();

	/*!
	 * Compute all per blob features, including regionprops, pixel intensity, gradient magnitude, Sobel, and Canny pixels
	 */
	void doAll();

	void doIntensity();

	/* Functions used to manage data transfers among CPU and GPU*/
	bool uploadImageToGPU();
	void releaseGPUImage();

	bool uploadImageMaskToGPU();
	void releaseGPUMask();


	/*!
	 *  Calculation of the co-occurence matrix
	 *  Description: Cx,y(i,j) = Sum_i Sum_j {1, if I(i,j) = i and I(i,j+1) = j; 0, otherwise
	 *
	 * 	Details : Calculated using the grayscale image, and as
	 * 	Matlab reduces it to 8 levels. Output is the same as Matlab default
	 */
	void printCoocMatrix(unsigned int angle=ANGLE_0);
	void doCoocMatrix(unsigned int angle=ANGLE_0);
	void doCoocMatrixGPU(unsigned int angle=ANGLE_0);

	/* Features based on co-occurrence matrix */

	/*!
	 * Inertia = Sum_i Sum_j (i-j)^2 p(i,j). For all calculation C(i,j)
	 * is normalized, generating p(i,j) = C(i,j)/Sum_i Sum_j C(i,j)
	 */
	double inertiaFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Energy = Sum_i Sum_j p(i,j)^2
	 */
	double energyFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*
	 * Entropy = Sum_i Sum_j p(i,j)log_2(p(i,j)),
	 *
	 * Details: Whether p(i,j) equal to 0, do not consider the index i,j to avoid nan results on calculation
	 */
	double entropyFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Homogeneity = Sum_i Sum_j (1/(1+(i-j)^2)) p(i,j)
	 */
	double homogeneityFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Maximum Probability = max_i,j p(i,j)
	 */
	double maximumProbabilityFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Cluster Shade = Sum_i Sum_j (k-M_x + j -M_y)^3 * p(i,j).
	 *
	 * k is distance among pixels when building the Cooc. Matrix.
	 * M_x = Sum_i Sum_j i*p(i,j).
	 * M_y = Sum_i Sum_j j*p(i,j)
	 */
	double clusterShadeFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Cluster Prominence = Sum_i Sum_j (k-M_x + j -M_y)^4 * p(i,j). k = distance among pixels when building the Cooc. Matrix.
	 *
	 * M_x = Sum_i Sum_j i*p(i,j).
	 * M_y = Sum_i Sum_j j*p(i,j)
	 */
	double clusterProminenceFromCoocMatrix(unsigned int angle=ANGLE_0, unsigned int procType=CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/* Calculation of Intensity statistics */
	double calcMeanIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	double calcStdIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMedianIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMinIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMaxIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcFirstQuartileIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcSecondQuartileIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcThirdQuartileIntensity(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);

	/* Calculation of Gradient statistics */
	double calcMeanGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	double calcStdGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMedianGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMinGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcMaxGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcFirstQuartileGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcSecondQuartileGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);
	int calcThirdQuartileGradientMagnitude(bool useMask=false, int procType=CPU, bool reuseItermediaryResults=true, int gpuId=0);

	/* Calculate Canny and Sobel pixels */
	int calcCannyArea(int procType=CPU, double lowThresh=0, double highThresh=255, int apertureSize = 3, int gpuId=0);
	int calcSobelArea(int procType=CPU, int xorder=1, int yorder=1, int apertureSize=7, bool useMask = false, int gpuId=0);

};

#endif /* REGIONALMORPHOLOGYANALYSIS_H_ */
