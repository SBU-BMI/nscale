/*
 * ObjFeatures.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#include "ObjFeatures.h"
#include "opencv2/gpu/gpu.hpp"
//#include "opencv2/gpu/devmem2d.hpp"

#if defined(WITH_CUDA)

#include "opencv2/gpu/stream_accessor.hpp"
#include "cuda/hist-ops.cuh"
#include "cuda/gradient.cuh"
#include "cuda/features.cuh"

#endif

namespace nscale{
namespace gpu{

using namespace cv::gpu;

#if !defined (WITH_CUDA)
float* ObjFeatures::intensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){	
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* ObjFeatures::gradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform");}

float* ObjFeatures::cannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& labeledMask, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream) {
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform");}

float* ObjFeatures::cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* ObjFeatures::cytoGradientFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
	CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

float* ObjFeatures::cytoCannyFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& GrayImage, cv::gpu::Stream& stream){
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
float* ObjFeatures::cytoIntensityFeatures(const int* boundingBoxesInfo, int compCount, const cv::gpu::GpuMat& grayImage, cv::gpu::Stream& stream){

	int* g_hist = nscale::gpu::calcHist256CytoBBCaller(grayImage, (int*)boundingBoxesInfo, compCount, StreamAccessor::getStream(stream) );

	float* g_features = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount * nscale::ObjFeatures::N_INTENSITY_FEATURES );

	nscale::gpu::calcFeaturesFromHist256Caller(g_hist,compCount, g_features, StreamAccessor::getStream(stream));

	// download hist for debugging only
//	int* h_temp_host = (int*) malloc( sizeof(int) * 256 );
//	nscale::gpu::cudaDownloadCaller(h_temp_host, g_hist, sizeof(int)*256);
//	for(int j = 0; j < 256; j++){
//			printf("gpu_hist[%d]=%d\n", j%256, h_temp_host[j]);
//	}
//	free(h_temp_host);

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
	cv::gpu::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

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
	cv::gpu::Canny(grayImage, cannyRes, 70.0, 90.0, 5);

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

/****************Written by Salil Deosthale*******************/
int *ObjFeatures::calculateArea(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask, cv::gpu::Stream& stream)
{
	//Create a pointer to the area on the GPU
	//std::cout << "Allocating space on the GPU" << std::endl;
	int *gArea = (int *)nscale::gpu::cudaMallocCaller(sizeof(int) * compCount);
	
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));


	//Call the AreaCaller function which calls the area kernel
	//std::cout << "Calling Area caller" << std::endl;
	AreaCaller((int *)boundingBoxesInfo , compCount, labeledMask , gArea, StreamAccessor::getStream(stream));

	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("gpu computation took %lld\n", t2-t0);

	//allocate space on the cpu for stroring the area features
	//std::cout << "Allocating space for area on the cpu" << std::endl;
	int *cArea = (int *)malloc(sizeof(int) * compCount);
	
	//Download the features to CPU
	//std::cout << "Downloading features from GPU to CPU" << std::endl;
	nscale::gpu::cudaDownloadCaller(cArea,gArea,sizeof(int) * compCount);
	
	//Free the area array created on the GPU
	//std::cout << "Freeing memory on the GPU" << std::endl;
	nscale::gpu::cudaFreeCaller(gArea);
	
	return cArea;
}

float *ObjFeatures::calculatePerimeter(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask , cv::gpu::Stream& stream)
{
	//Create a pointer to the perimeter on the GPU
	//std::cout << "Allocating space on the GPU" << std::endl;
	float *gPerimeter = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);

	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	//Call the perimeter caller function which calls the perimeter kernel
	//std::cout << "Calling perimeter caller" << std::endl;
	PerimeterCaller((int *)boundingBoxesInfo, compCount, labeledMask, gPerimeter, StreamAccessor::getStream(stream));
	cudaDeviceSynchronize();

	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("gpu computation took %lld\n", t2-t0);

	
	//Allocate area on the cpu for storing the perimeter
	float *cPerimeter = (float *) malloc(sizeof(float) * compCount);
	
	//Download the features to the CPU
	//std::cout << "Downloading features to the CPU" << std::endl;
	nscale::gpu::cudaDownloadCaller(cPerimeter , gPerimeter , sizeof(float) * compCount);
	
	
	//Free the area created on the gpu
	//std::cout << "Freeing memory on the GPU" << std::endl;	
	nscale::gpu::cudaFreeCaller(gPerimeter);
	
	return cPerimeter;
}

void ObjFeatures::calculateEllipse(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask , int* areaRes, float* &majorAxis , float* &minorAxis , float* &ecc, cv::gpu::Stream& stream)
{
	//the above pointers majorAxis, minorAxis and ecc live on the CPU. allocate their GPU counterparts
	float *gmajorAxis = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gminorAxis = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gEcc = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	
	//Time the kernel call
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	//Kernel call
	EllipseCaller((int*)boundingBoxesInfo , compCount ,labeledMask ,areaRes ,gmajorAxis ,gminorAxis , gEcc, StreamAccessor::getStream(stream));
	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("gpu computation took %lld\n", t2-t0);
	
	
	//The cpu pointers majorAxis, minorAxis and ecc have not been allocated yet. ALLOCATE THEM PUNY MORTAL!!!!!!!
	majorAxis = (float *) malloc(sizeof(float) * compCount);
	minorAxis = (float *) malloc(sizeof(float) * compCount);
	ecc = (float *) malloc(sizeof(float) * compCount);
	

	nscale::gpu::cudaDownloadCaller(majorAxis , gmajorAxis , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(minorAxis , gminorAxis , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(ecc , gEcc , sizeof(float) * compCount);
	
	
	//free the area created on the GPU
	nscale::gpu::cudaFreeCaller(gmajorAxis);
	nscale::gpu::cudaFreeCaller(gminorAxis);
	nscale::gpu::cudaFreeCaller(gEcc);
}

float *ObjFeatures::calculateExtentRatio(const int* boundingBoxesInfo , const int compCount , const int* areaRes , cv::gpu::Stream& stream)
{
	//Allocate some memory for the extent ratio on the gpu
	float *gExtentRatio = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	//Allocate some memory for the extent ratio on the cpu
	float *cExtentRatio = (float *)malloc(sizeof(float) * compCount);
	
	//Time the kernel call
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	//Kernel call
	ExtentRatioCaller( (int*)boundingBoxesInfo , compCount , areaRes , gExtentRatio, StreamAccessor::getStream(stream));
	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("gpu computation took %lld\n", t2-t0);
	
	//Download the data from the gpu
	nscale::gpu::cudaDownloadCaller(cExtentRatio , gExtentRatio , sizeof(float) * compCount);
	
	//free the memory on the GPU
	nscale::gpu::cudaFreeCaller(gExtentRatio);
		
	return cExtentRatio;
	
}

float *ObjFeatures::calculateCircularity(const int compCount , const int *areaRes , const float *perimeterRes , cv::gpu::Stream& stream)
{
	//Allocate some memory for the circularity on the gpu
	float *gCircularity = (float *) nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	//Allocate some memory for the circularity on the cpu
	float *cCircularity = (float *) malloc(sizeof(float) * compCount);
	
	//Time the kernel call
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	//Kernel call
	CircularityCaller( compCount , areaRes , perimeterRes , gCircularity , StreamAccessor::getStream(stream));
	
	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("gpu computation took %lld\n", t2-t0);
	
	//Download the data from the GPU
	nscale::gpu::cudaDownloadCaller(cCircularity , gCircularity , sizeof(float) * compCount);
	
	//free the memory on the GPU
	nscale::gpu::cudaFreeCaller(gCircularity);
	
	return cCircularity;
}

void ObjFeatures::calculateAllFeatures(const int* boundingBoxesInfo , int compCount , const cv::gpu::GpuMat& labeledMask, int* &areaRes , float* &perimeterRes , float* &majorAxis , float* &minorAxis , float* &ecc, float* &extent_ratio , float* &circ, cv::gpu::Stream& stream)
{
	
	//Allocate some memory on the GPU
	int *gArea = (int *)nscale::gpu::cudaMallocCaller(sizeof(int) * compCount);
	float *gPerimeter = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gmajorAxis = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gminorAxis = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gEcc = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *g_extent_ratio = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *g_circ = (float *)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);

	//Time the kernel call
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	BigFeaturesCaller((int*) boundingBoxesInfo , compCount , labeledMask , gArea , gPerimeter , gmajorAxis , gminorAxis , gEcc , StreamAccessor::getStream(stream));
	
	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	long long t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("\n\ngpu computation of BigFeatures took %lld\n", t2-t0);


	gettimeofday(&ts, NULL);
	t0 = (ts.tv_sec*1000000 + (ts.tv_usec));
	
	SmallFeaturesCaller((int*) boundingBoxesInfo , compCount , gArea , gPerimeter , g_extent_ratio , g_circ , StreamAccessor::getStream(stream));
	
	cudaDeviceSynchronize();
	gettimeofday(&ts, NULL);
	t2 = (ts.tv_sec*1000000 + (ts.tv_usec));
	printf("\n\ngpu computation of SmallFeatures took %lld\n", t2-t0);
	
	
	//Allocate memory on the CPU
	areaRes = (int *)malloc(sizeof(int) * compCount);
	perimeterRes = (float *)malloc(sizeof(float) * compCount);
	majorAxis = (float *) malloc(sizeof(float) * compCount);
	minorAxis = (float *) malloc(sizeof(float) * compCount);
	ecc = (float *) malloc(sizeof(float) * compCount);
	extent_ratio = (float *)malloc(sizeof(float) * compCount);
	circ = (float *)malloc(sizeof(float) * compCount);
	
	//Download the data from the GPU
	nscale::gpu::cudaDownloadCaller(areaRes,gArea,sizeof(int) * compCount);
	nscale::gpu::cudaDownloadCaller(perimeterRes , gPerimeter , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(majorAxis , gmajorAxis , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(minorAxis , gminorAxis , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(ecc , gEcc , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(extent_ratio , g_extent_ratio , sizeof(float) * compCount);
	nscale::gpu::cudaDownloadCaller(circ , g_circ , sizeof(float) * compCount);
	
	//Free the memory on the GPU
	nscale::gpu::cudaFreeCaller(gArea);
	nscale::gpu::cudaFreeCaller(gPerimeter);
	nscale::gpu::cudaFreeCaller(gmajorAxis);
	nscale::gpu::cudaFreeCaller(gminorAxis);
	nscale::gpu::cudaFreeCaller(gEcc);
	nscale::gpu::cudaFreeCaller(g_extent_ratio);
	nscale::gpu::cudaFreeCaller(g_circ);
	
	
}



#endif

}}

