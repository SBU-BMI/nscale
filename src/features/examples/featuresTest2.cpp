#include "opencv2/opencv.hpp"

#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif 

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <dirent.h>
#include <fstream>

#include "ObjFeatures.h"
#include "CytoplasmCalc.h"
#include "HistologicalEntities.h"
#include "Logger.h"

using namespace cv;
#ifdef WITH_CUDA
using namespace cv::gpu;
#endif


int main (int argc, char **argv){

	std::cout << "Usage: <input image>"<< std::endl;
	cv::Mat inputImg = cv::imread(argv[1], -1);
	cv::Mat labeledImage;
	int compCount;
	int *bbox;
	uint64_t t1, t0, tinit, tend;
	uint64_t cpuStart , cpuEnd;

	nscale::HistologicalEntities::segmentNuclei(inputImg, labeledImage, compCount, bbox);

	
	cpuStart = cci::common::event::timestampInUS();
	double *cpuPerimeter = nscale::ObjFeatures::perimeter((const int*)bbox,compCount,labeledImage);
	cpuEnd = cci::common::event::timestampInUS();
	std::cout << "CPU Perimeter = " << cpuEnd - cpuStart << "\n" << std::endl;


	cpuStart = cci::common::event::timestampInUS();
	int *cpuArea = nscale::ObjFeatures::area((const int*)bbox , compCount , labeledImage);
	cpuEnd = cci::common::event::timestampInUS();
	std::cout << "CPU Area = " << cpuEnd - cpuStart << "\n" << std::endl;



	double *cpuMajorAxis , *cpuMinorAxis , *cpuEccentricity;
	cpuStart = cci::common::event::timestampInUS();
	nscale::ObjFeatures::ellipse((const int*)bbox,(const int*)cpuArea, compCount , labeledImage, cpuMajorAxis, cpuMinorAxis, cpuEccentricity);
	cpuEnd = cci::common::event::timestampInUS();
	std::cout << "CPU Ellipse = " << cpuEnd - cpuStart << "\n" << std::endl;


	cpuStart = cci::common::event::timestampInUS();
	double *cpuExtentRatio = nscale::ObjFeatures::extent_ratio((const int*)bbox , (const int)compCount , (const int *)cpuArea);
	cpuEnd = cci::common::event::timestampInUS();
	std::cout << "CPU Extent Ratio = " << cpuEnd - cpuStart << "\n" << std::endl;
	

	cpuStart = cci::common::event::timestampInUS()	;
	double *cpuCircularity = nscale::ObjFeatures::circularity(compCount , cpuArea , cpuPerimeter);
	cpuEnd = cci::common::event::timestampInUS();
	std::cout << "CPU Circularity = " << cpuEnd - cpuStart << "\n" << std::endl;

	

#if defined(WITH_CUDA)
	Stream stream;
	GpuMat g_labeledImage;

	int*g_bbox = (int*)nscale::gpu::cudaMallocCaller(sizeof(int) * 5 * compCount);
	nscale::gpu::cudaUploadCaller(g_bbox, bbox, sizeof(int) * 5 * compCount);
	stream.enqueueUpload(labeledImage, g_labeledImage);
	stream.waitForCompletion();

	
	/*******************************************MORPHOLOGICAL FEATURE CALCULATION ON THE GPU*******************************************/
	//First declare the pointers to the various big features
	int *gpuArea = (int*)nscale::gpu::cudaMallocCaller(sizeof(int) * compCount);
	float *gpuPerimeter = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gpuMajorAxis;
	float *gpuMinorAxis;
	float *gpuEccentricity;
	float *gpuExtentRatio;
	float *gpuCircularity;
	


	t0 = cci::common::event::timestampInUS();
	int* area = nscale::gpu::ObjFeatures::calculateArea(g_bbox , compCount , g_labeledImage , stream);
	//stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU Area  = "<< t1-t0 << "\n" << std::endl;
	nscale::gpu::cudaUploadCaller(gpuArea, area, sizeof(int) * compCount);
	free(area);

	t0 = cci::common::event::timestampInUS();
	float *perimeter = nscale::gpu::ObjFeatures::calculatePerimeter(g_bbox , compCount , g_labeledImage , stream);
	//stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU Perimeter  = "<< t1-t0 << "\n" << std::endl;
	nscale::gpu::cudaUploadCaller(gpuPerimeter, perimeter, sizeof(int) * compCount);
	free(perimeter);

	t0 = cci::common::event::timestampInUS();
	gpuExtentRatio = nscale::gpu::ObjFeatures::calculateExtentRatio (g_bbox , compCount , gpuArea , stream);
	//stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU Extent Ratio  = "<< t1-t0 << "\n" << std::endl;

	t0 = cci::common::event::timestampInUS();
	gpuCircularity = nscale::gpu::ObjFeatures::calculateCircularity(compCount , gpuArea , gpuPerimeter , stream);
	//stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU Circularity  = "<< t1-t0 << "\n" << std::endl;
	
	t0 = cci::common::event::timestampInUS();
	nscale::gpu::ObjFeatures::calculateEllipse(g_bbox , compCount , g_labeledImage , gpuArea , gpuMajorAxis , gpuMinorAxis , gpuEccentricity , stream);
	//stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU Ellipse  = "<< t1-t0 << "\n" << std::endl;


	free(gpuCircularity);
	free(gpuExtentRatio);
	nscale::gpu::cudaFreeCaller(gpuArea);
	nscale::gpu::cudaFreeCaller(gpuPerimeter);
	free(gpuMajorAxis);
	free(gpuMinorAxis);
	free(gpuEccentricity);
	
	t0 = cci::common::event::timestampInUS();
	nscale::gpu::ObjFeatures::calculateAllFeatures(g_bbox , compCount , g_labeledImage, gpuArea , gpuPerimeter , gpuMajorAxis , gpuMinorAxis , gpuEccentricity, gpuExtentRatio , gpuCircularity, stream);
	t1 = cci::common::event::timestampInUS();
	std::cout << "GPU all morpho features  = "<< t1-t0 << "\n" << std::endl;

	free(gpuCircularity);
	free(gpuExtentRatio);
	free(gpuArea);
	free(gpuPerimeter);
	free(gpuMajorAxis);
	free(gpuMinorAxis);
	free(gpuEccentricity);

	g_labeledImage.release();

	nscale::gpu::cudaFreeCaller(g_bbox);

#endif

	free(cpuCircularity);
	free(cpuExtentRatio);
	free(cpuArea);
	free(cpuPerimeter);
	free(cpuMajorAxis);
	free(cpuMinorAxis);
	free(cpuEccentricity);

	free(bbox);

	return 0;

}


