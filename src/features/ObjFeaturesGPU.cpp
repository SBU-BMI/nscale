/*
 * ObjFeatures.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#include "ObjFeatures.h"
#include "opencv2/gpu/gpu.hpp"
//#include "opencv2/gpu/devmem2d.hpp"

#if defined(HAVE_CUDA)

#include "opencv2/gpu/stream_accessor.hpp"
#include "cuda/hist-ops.cuh"
#include "cuda/gradient.cuh"

#endif

namespace nscale{
namespace gpu{

using namespace cv::gpu;

#if !defined (HAVE_CUDA)
float* ObjFeatures::intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){	
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* ObjFeatures::gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform");}

float* ObjFeatures::cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream) {
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform");}

float* cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

#else
float* ObjFeatures::intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask,
		const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){

	int* g_hist = nscale::gpu::calcHist256Caller(labeledMask, grayImage, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream) );

	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_INTENSITY_FEATURES );

	nscale::gpu::calcFeaturesFromHist256Caller(g_hist,compCount, g_features, StreamAccessor::getStream(stream));

	// alloc space in the CPU to store features
	float *cpu_features = (float*)malloc(sizeof(float) * nscale::ObjFeatures::N_INTENSITY_FEATURES * compCount);

	// Download features to cpu.
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)* nscale::ObjFeatures::N_INTENSITY_FEATURES * compCount);

	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	return cpu_features;

}
float* cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){

	int* g_hist = nscale::gpu::calcHist256CytoBBCaller(grayImage, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream) );

	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_INTENSITY_FEATURES );

	nscale::gpu::calcFeaturesFromHist256Caller(g_hist,compCount, g_features, StreamAccessor::getStream(stream));

	// alloc space in the CPU to store features
	float *cpu_features = (float*)malloc(sizeof(float) * nscale::ObjFeatures::N_INTENSITY_FEATURES * compCount);

	// Download features to cpu.
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)* nscale::ObjFeatures::N_INTENSITY_FEATURES * compCount);

	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	return cpu_features;
}







float* ObjFeatures::gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){
	// First calculate the gradient of the input image
	cv::gpu::GpuMat g_gradient(grayImage.size(), CV_32FC1);

	// in x direction, and accumulate square result in g_gradient
	nscale::gpu::xDerivativeSquareCaller(grayImage.rows, grayImage.cols, grayImage, g_gradient, StreamAccessor::getStream(stream));

	// in y direction, and accumulate the square result again in g_gradient
	nscale::gpu::yDerivativeSquareAccCaller(grayImage.rows, grayImage.cols, grayImage, g_gradient, StreamAccessor::getStream(stream));

	// Convert float gradient to unsigned char [0:255]
	cv::gpu::GpuMat g_gradient_grayscale(grayImage.size(), CV_8UC1);

	nscale::gpu::floatToUcharCaller(grayImage.rows, grayImage.cols,  g_gradient, g_gradient_grayscale, StreamAccessor::getStream(stream));

	// Calculate the histogram for each object
	int* g_hist = nscale::gpu::calcHist256Caller(labeledMask, g_gradient_grayscale, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream));

//	// download hist for debugging only
//	int* h_temp_host = (int*) malloc( sizeof(int) * 256 * 2 );
//	nscale::gpu::cudaDownloadCaller(h_temp_host, g_hist, sizeof(int)*256*2);
//	for(int j = 0; j < 512; j++){
//			printf("gpu_hist[%d]=%d\n", j%256, h_temp_host[j]);
//	}


	// Allocate space to store feature in GPU
	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_GRADIENT_FEATURES );

	// Calculate gradient features
	nscale::gpu::calcGradFeaturesFromHist256Caller(g_hist, compCount, g_features, StreamAccessor::getStream(stream));

	// Alloc host memory to transfer feature to CPU
	float* cpu_features = (float*)malloc(sizeof(float)*nscale::ObjFeatures::N_GRADIENT_FEATURES * compCount);

	// Download features
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)*nscale::ObjFeatures::N_GRADIENT_FEATURES*compCount);

	// Release histogram and features memory allocated on GPU
	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	// Release temporary data
	g_gradient.release();
	g_gradient_grayscale.release();

	// Retrieve in CPU memory features
	return cpu_features;
//	return NULL;
};



float* ObjFeatures::cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){
	// First calculate the gradient of the input image
	cv::gpu::GpuMat g_gradient(grayImage.size(), CV_32FC1);

	// in x direction, and accumulate square result in g_gradient
	nscale::gpu::xDerivativeSquareCaller(grayImage.rows, grayImage.cols, grayImage, g_gradient, StreamAccessor::getStream(stream));

	// in y direction, and accumulate the square result again in g_gradient
	nscale::gpu::yDerivativeSquareAccCaller(grayImage.rows, grayImage.cols, grayImage, g_gradient, StreamAccessor::getStream(stream));

	// Convert float gradient to unsigned char [0:255]
	cv::gpu::GpuMat g_gradient_grayscale(grayImage.size(), CV_8UC1);

	nscale::gpu::floatToUcharCaller(grayImage.rows, grayImage.cols,  g_gradient, g_gradient_grayscale, StreamAccessor::getStream(stream));

	// Calculate the histogram for each object
	int* g_hist = nscale::gpu::calcHist256CytoBBCaller(g_gradient_grayscale, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream));

//	// download hist for debugging only
//	int* h_temp_host = (int*) malloc( sizeof(int) * 256 * 2 );
//	nscale::gpu::cudaDownloadCaller(h_temp_host, g_hist, sizeof(int)*256*2);
//	for(int j = 0; j < 512; j++){
//			printf("gpu_hist[%d]=%d\n", j%256, h_temp_host[j]);
//	}


	// Allocate space to store feature in GPU
	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_GRADIENT_FEATURES );

	// Calculate gradient features
	nscale::gpu::calcGradFeaturesFromHist256Caller(g_hist, compCount, g_features, StreamAccessor::getStream(stream));

	// Alloc host memory to transfer feature to CPU
	float* cpu_features = (float*)malloc(sizeof(float)*nscale::ObjFeatures::N_GRADIENT_FEATURES * compCount);

	// Download features
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)*nscale::ObjFeatures::N_GRADIENT_FEATURES*compCount);

	// Release histogram and features memory allocated on GPU
	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	// Release temporary data
	g_gradient.release();
	g_gradient_grayscale.release();

	// Retrieve in CPU memory features
	return cpu_features;
};




float* ObjFeatures::cannyFeatures(const int* boundingBoxesInfo, int compCount,
		const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream) {

	cv::gpu::GpuMat cannyRes(grayImage.size(), grayImage.type());
//	cv::gpu::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

	// Calculate histogram from Canny results for each object
	int* g_hist = nscale::gpu::calcHist256Caller(labeledMask, cannyRes, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream) );

	// Alloc memory to store canny features results on GPU
	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_CANNY_FEATURES );

	// Calculate the features
	nscale::gpu::calcCannyFeaturesFromHist256Caller(g_hist, compCount, g_features, StreamAccessor::getStream(stream) );

	// alloc space in the CPU to store features
	float *cpu_features = (float*)malloc(sizeof(float) * nscale::ObjFeatures::N_CANNY_FEATURES * compCount);

	// Download features to CPU.
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)* nscale::ObjFeatures::N_CANNY_FEATURES * compCount);

	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	// release Canny matrix
	cannyRes.release();

	// return features
	return cpu_features;

}


float* ObjFeatures::cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream) {

	cv::gpu::GpuMat cannyRes(grayImage.size(), grayImage.type());
//	cv::gpu::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

	// Calculate histogram from Canny results for each object
	int* g_hist = nscale::gpu::calcHist256CytoBBCaller(cannyRes, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream) );

	// Alloc memory to store canny features results on GPU
	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_CANNY_FEATURES );

	// Calculate the features
	nscale::gpu::calcCannyFeaturesFromHist256Caller(g_hist, compCount, g_features, StreamAccessor::getStream(stream) );

	// alloc space in the CPU to store features
	float *cpu_features = (float*)malloc(sizeof(float) * nscale::ObjFeatures::N_CANNY_FEATURES * compCount);

	// Download features to CPU.
	nscale::gpu::cudaDownloadCaller(cpu_features, g_features, sizeof(float)* nscale::ObjFeatures::N_CANNY_FEATURES * compCount);

	nscale::gpu::cudaFreeCaller(g_hist);
	nscale::gpu::cudaFreeCaller(g_features);

	// release Canny matrix
	cannyRes.release();

	// return features
	return cpu_features;

}

#endif

}}

