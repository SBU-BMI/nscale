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

#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif 

#ifdef _MSC_VER
#include "time_win.h"
#else
#include <sys/time.h>
#endif
#include "Operators.h"

#include "../segment/HistologicalEntities.h"
#include "../segment/PixelOperations.h"
#include "../segment/MorphologicOperations.h"
#include "FileUtils.h"
#include "ObjFeatures.h"
#include "ConnComponents.h"


namespace nscale{

class ObjFeatures {
public:
	//Geometric features
	static int* area(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask);
	static void ellipse(const int* boundingBoxesInfo,const int* areaRes, const int compCount , const cv::Mat& labeledMask, double* &majorAxis, double* &minorAxis, double* &ecc);
	static double *extent_ratio(const int* boundingBoxesInfo, const int compCount, const int* areaRes);
	static double* perimeter(const int* boundingBoxesInfo, const int compCount,const cv::Mat& labeledMask);
	static double* circularity(const int compCount, const int* areaRes, const double* perimeter);


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
	static float* calcAvgFeatures(cv::Mat& bgr, const cv::Mat& mask);

};

#ifdef WITH_CUDA
namespace gpu{
class ObjFeatures {
public:
	static float* intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream);

	static float* cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);
	static float* cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream);

	static int *calculateArea(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask, cv::gpu::Stream& stream);
	static float *calculatePerimeter(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask , cv::gpu::Stream& stream);
	static void calculateEllipse(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask , int* areaRes, float* &majorAxis , float* &minorAxis , float* &ecc, cv::gpu::Stream& stream);
	static float *calculateExtentRatio(const int *boundingBoxesInfo , const int compCount , const int *areaRes, cv::gpu::Stream& stream);
	static float *calculateCircularity(const int compCount , const int *areaRes , const float *perimeterRes , cv::gpu::Stream& stream);
	static void calculateAllFeatures(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask, int* &areaRes , float* &perimeterRes , float* &majorAxis , float* &minorAxis , float* &ecc, float* &extent_ratio , float* &circ, cv::gpu::Stream& stream);

};

// Auxiliar functions used to access cuda api from c++/c code
void cudaFreeCaller(void *data_ptr);
void *cudaMallocCaller(int size);
void cudaUploadCaller(void *dest, void *source, int size);
void cudaDownloadCaller(void *dest, void *source, int size);

}
#endif 

}// end nscale
#endif /* OBJFEATURES_H_ */
