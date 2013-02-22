/*
 * ObjFeatures.h
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#ifndef OBJFEATURES_H_
#define OBJFEATURES_H_

// Includes to use opencv2/GPU
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>
#include "Operators.h"

#include "HistologicalEntities.h"
#include "PixelOperations.h"
#include "MorphologicOperations.h"
#include "FileUtils.h"
#include "ObjFeatures.h"
#include "ConnComponents.h"


namespace nscale{

class ObjFeatures {
public:
	static int* area(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask);
	void ellipse(const int* boundingBoxesInfo,const int* areaRes, const int compCount , const cv::Mat& labeledMask, double *majorAxis, double *minorAxis, double *ecc);


	static const int N_INTENSITY_FEATURES=8;
	static const int N_GRADIENT_FEATURES=6;
	static const int N_CANNY_FEATURES=2;

	static float* intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage);
	static float* gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage);
	static float* cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask, const cv::Mat& grayImage);

	static float* cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage);
	static float* cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage);
	static float* cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::Mat& grayImage);

	static void calcFeatures(cv::Mat& bgr, const cv::Mat& mask);
};

namespace gpu{
class ObjFeatures {
public:
	static float* intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream);

	static float* cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);

};

// Auxiliar functions used to access cuda api from c++/c code
void cudaFreeCaller(void *data_ptr);
void *cudaMallocCaller(int size);
void cudaUploadCaller(void *dest, void *source, int size);
void cudaDownloadCaller(void *dest, void *source, int size);

}
}// end nscale
#endif /* OBJFEATURES_H_ */
