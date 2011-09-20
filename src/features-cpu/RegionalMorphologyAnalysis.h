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
#include "Constants.h"

#include <iomanip>
#include <sys/time.h>

// Includes used by opencv
#include "highgui.h"
#include "cv.h"
#include "cxcore.h"

// Includes to use opencv2/GPU
#include "opencv2/opencv.hpp"
#ifdef USE_GPU
#include "opencv2/gpu/gpu.hpp"
#endif


using namespace std;
using namespace cv;

class ROI{
public:
	int x;
	int y;
	int width;
	int height;
	ROI(int x, int y, int width, int height){
		this->x = x; 
		this->y = y;
		this->width = width;
		this->height = height;
	};
	~ROI(){};
};

class RegionalMorphologyAnalysis {
private:

	//! Vector holding the blob identified in the input image
	vector<Blob *> internal_blobs;

	//! Pointer to a in-memory copy of the input image
	IplImage *originalImage;

	//! Pointer to a in-memory copy of the input image mask
	IplImage *originalImageMask;

	//! Says whether the pointers are images: true, or just image headers:false
	bool isImage;

	//! Pointer to a in-memory mask for the bounding boxes containing nucleus found in origina image
	// (int)Blob1 offset | .. | (int)BlobN offset | (int)Blob1.x |(int)Blob1.y| (int)Blob1.width| (int)Blob1.height | (chars)Mask data | BlobN.x | BlobN.y | BlobN.width | BlobN.height| Mask data
	Mat *originalImageMaskNucleusBoxes;

	//! Pointer to a in-memory mask for the bounding boxes containing nucleus found in origina image
	// (int)Blob1 offset | .. | (int)BlobN offset | (int)Blob1.x |(int)Blob1.y| (int)Blob1.width| (int)Blob1.height | (chars)Mask data | BlobN.x | BlobN.y | BlobN.width | BlobN.height| Mask data
//	char *blobsMaskAllocatedMemory;

	//! Size of the mask containing the nucleus bounding boxes
	int blobsMaskAllocatedMemorySize;

	//! Auxiliary function used to print information stored in the bounding box masks
	void printBlobsAllocatedInfo();

#ifdef	USE_GPU
	//! Pointer to a copy of the input image in the GPU memory.
	// If not NULL its pointer refers to the copy in the GPU main memory
	cv::gpu::GpuMat *originalImageGPU;

	//! Pointer to a copy of the input image mask in the GPU memory.
	// If not NULL its pointer refers to the copy in the GPU main memory
	cv::gpu::GpuMat *originalImageMaskGPU;

	cv::gpu::GpuMat *originalImageMaskNucleusBoxesGPU;
#endif


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
	unsigned int *calcGradientHistogram(bool useMask, int procType, bool reuseItermediaryResults, ROI * roi = NULL, int gpuId = 0);
	unsigned int *calcIntensityHistogram(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi = NULL, int gpuId=0);


public:
	// Images are read from disk using the path given as parameters
	RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName);


	// Image stored in memory is given as parameter, and they are not copied but we only point to those images. 
	// So, these images should not be deleted or any call to the RegionalMorphologyAnalysis will to fail.
	RegionalMorphologyAnalysis(IplImage *originalImageMask, IplImage *originalImage);
	

	virtual ~RegionalMorphologyAnalysis();

	void printStats();
	unsigned int *getIntensityHist(){
		return intensity_hist;
	};
	unsigned int *getGradHist(){
		return gradient_hist;
	};
	unsigned int *getCoocMatrix(int angle){
		return coocMatrix[angle];
	};

	int getImgWidth(){
		return originalImage->width;
	}

	int getImgHeight(){
		return originalImage->height;
	}
	/*!
	 * Calculates Haralick features derived from coocMatrix for each blob.
	 */
	void doCoocPropsBlob(vector<vector<float> > &haralickFeatures, unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0, char *gpuTempData=NULL);

	/*!
	 * Computes Intensity features from each blob found in input image.
	 */

	void doIntensityBlob(vector<vector<float> > &intensityFeatures,unsigned int procType=Constant::CPU, unsigned int gpuId=0);
	/*!
	 * Computes Gradient features from each blob found in input image.
	 */
	void doGradientBlob(vector<vector<float> > &gradientFeatures, unsigned int procType=Constant::CPU, unsigned int gpuId=0);

	/*!
	 * Computes Morphometry features from each blob found in input image.
	 */
	void doMorphometryFeatures(vector<vector<float> > &morphoFeatures);

	/*!
	 * Compute all per blob features, including regionprops, pixel intensity, gradient magnitude, Sobel, and Canny pixels
	 */
	void doAll();



	/* Functions used to manage data transfers among CPU and GPU*/
	bool uploadImageToGPU();
	void releaseGPUImage();

	bool uploadImageMaskToGPU();
	void releaseGPUMask();

	bool uploadImageMaskNucleusToGPU();
	void releaseImageMaskNucleusToGPU();

	/*!
	 *  Calculation of the co-occurence matrix
	 *  Description: Cx,y(i,j) = Sum_i Sum_j {1, if I(i,j) = i and I(i,j+1) = j; 0, otherwise
	 *
	 * 	Details : Calculated using the grayscale image, and as
	 * 	Matlab reduces it to 8 levels. Output is the same as Matlab default
	 */
	void printCoocMatrix(unsigned int angle=Constant::ANGLE_0);
	void doCoocMatrix(unsigned int angle=Constant::ANGLE_0);
	void doCoocMatrixGPU(unsigned int angle=Constant::ANGLE_0);

	/* Features based on co-occurrence matrix */

	/*!
	 * Inertia = Sum_i Sum_j (i-j)^2 p(i,j). For all calculation C(i,j)
	 * is normalized, generating p(i,j) = C(i,j)/Sum_i Sum_j C(i,j)
	 */
	double inertiaFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Energy = Sum_i Sum_j p(i,j)^2
	 */
	double energyFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*
	 * Entropy = Sum_i Sum_j p(i,j)log_2(p(i,j)),
	 *
	 * Details: Whether p(i,j) equal to 0, do not consider the index i,j to avoid nan results on calculation
	 */
	double entropyFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Homogeneity = Sum_i Sum_j (1/(1+(i-j)^2)) p(i,j)
	 */
	double homogeneityFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Maximum Probability = max_i,j p(i,j)
	 */
	double maximumProbabilityFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Cluster Shade = Sum_i Sum_j (k-M_x + j -M_y)^3 * p(i,j).
	 *
	 * k is distance among pixels when building the Cooc. Matrix.
	 * M_x = Sum_i Sum_j i*p(i,j).
	 * M_y = Sum_i Sum_j j*p(i,j)
	 */
	double clusterShadeFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/*!
	 * Cluster Prominence = Sum_i Sum_j (k-M_x + j -M_y)^4 * p(i,j). k = distance among pixels when building the Cooc. Matrix.
	 *
	 * M_x = Sum_i Sum_j i*p(i,j).
	 * M_y = Sum_i Sum_j j*p(i,j)
	 */
	double clusterProminenceFromCoocMatrix(unsigned int angle=Constant::ANGLE_0, unsigned int procType=Constant::CPU, bool reuseItermediaryResults=true, unsigned int gpuId=0);

	/* Calculation of Intensity statistics */
	double calcMeanIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	double calcStdIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMedianIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMinIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMaxIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcFirstQuartileIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcSecondQuartileIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcThirdQuartileIntensity(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);

	/* Calculation of Gradient statistics */
	double calcMeanGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	double calcStdGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMedianGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMinGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcMaxGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcFirstQuartileGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcSecondQuartileGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);
	int calcThirdQuartileGradientMagnitude(bool useMask=false, int procType=Constant::CPU, bool reuseItermediaryResults=true, ROI *roi=NULL, int gpuId=0);

	/* Calculate Canny and Sobel pixels */
	int calcCannyArea(int procType=Constant::CPU, double lowThresh=0, double highThresh=255, int apertureSize = 3, ROI *roi=NULL, int gpuId=0);
	int calcSobelArea(int procType=Constant::CPU, int xorder=1, int yorder=1, int apertureSize=7, bool useMask = false, ROI *roi=NULL, int gpuId=0);

};

#endif /* REGIONALMORPHOLOGYANALYSIS_H_ */
