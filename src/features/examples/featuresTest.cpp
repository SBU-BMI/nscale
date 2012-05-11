#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <dirent.h>

#include "ObjFeatures.h"
#include "CytoplasmCalc.h"
#include "HistologicalEntities.h"

using namespace cv;
using namespace cv::gpu;


int main (int argc, char **argv){

	std::cout << "Usage: <input image>"<< std::endl;
	cv::Mat inputImg = cv::imread(argv[1], -1);
	cv::Mat labeledImage;
	int compCount;
	int *bbox;
	uint64_t t1, t0;

	nscale::HistologicalEntities::segmentNuclei(inputImg, labeledImage, compCount, bbox);
//	int* objsArea = nscale::ObjFeatures::area((const int *)bbox, compCount, labeledImage);
//	free(objsArea);

	t0 = cciutils::ClockGetTime();
	int* cytoplasmBB = nscale::CytoplasmCalc::calcCytoplasm(bbox, compCount, labeledImage);
	t1 = cciutils::ClockGetTime();
	std::cout << "CytoplasmMaskCalc = "<< t1-t0 <<std::endl;

	vector<cv::Mat> bgr;
	split(inputImg, bgr);
	t0 = cciutils::ClockGetTime();
	float* intensityFeatures = nscale::ObjFeatures::intensityFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cciutils::ClockGetTime();

	std::cout << "NucleiIntensityFeaturesTime = "<< t1-t0 <<std::endl;
	free(intensityFeatures);

	t0 = cciutils::ClockGetTime();
	float* cytoIntensityFeatures = nscale::ObjFeatures::cytoIntensityFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cciutils::ClockGetTime();

	std::cout << "CytoIntensityFeaturesTime = "<< t1-t0 <<std::endl;
	free(cytoIntensityFeatures);


	t0 = cciutils::ClockGetTime();
	float* h_gradientFeatures = nscale::ObjFeatures::gradientFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cciutils::ClockGetTime();
	std::cout << "NucleiGradientFeaturesTimeCPU = "<< t1-t0 <<std::endl;
	free(h_gradientFeatures);


	t0 = cciutils::ClockGetTime();
	float* h_cytoGradientFeatures = nscale::ObjFeatures::cytoGradientFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cciutils::ClockGetTime();
	std::cout << "CytoGradientFeaturesTimeCPU = "<< t1-t0 <<std::endl;
	free(h_cytoGradientFeatures);


	t0 = cciutils::ClockGetTime();
	float* h_cannyFeatures = nscale::ObjFeatures::cannyFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cciutils::ClockGetTime();
	std::cout << "NucleiCannyFeaturesTimeCPU = "<< t1-t0 <<std::endl;
	free(h_cannyFeatures);

	t0 = cciutils::ClockGetTime();
	float* h_cytoCannyFeatures = nscale::ObjFeatures::cytoCannyFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cciutils::ClockGetTime();
	std::cout << "CytoCannyFeaturesTimeCPU = "<< t1-t0 <<std::endl;
	free(h_cytoCannyFeatures);

#if defined(HAVE_CUDA)
	Stream stream;
	GpuMat g_labeledImage;
	GpuMat g_gray;

	int*g_bbox = (int*)nscale::gpu::cudaMallocCaller(sizeof(int) * 5 * compCount);
	nscale::gpu::cudaUploadCaller(g_bbox, bbox, sizeof(int) * 5 * compCount);

	stream.enqueueUpload(labeledImage, g_labeledImage);
	stream.enqueueUpload(bgr[0], g_gray);
	stream.waitForCompletion();


	t0 = cciutils::ClockGetTime();
	float* h_intensityFeatures = nscale::gpu::ObjFeatures::intensityFeatures(g_bbox,  compCount, g_labeledImage , g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "IntensityFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(h_intensityFeatures);

	t0 = cciutils::ClockGetTime();
	float* g_h_gradientFeatures = nscale::gpu::ObjFeatures::gradientFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	//= nscale::gpu::ObjFeatures::gradientFeatures(g_bbox,  compCount, g_labeledImage , g_gray, stream);
	t1 = cciutils::ClockGetTime();
//	for(int i = 0; i < 12; i++){
//		cout << "GradFeatureGPU id["<<i<<"] = "<< g_h_gradientFeatures[i]<<endl;
//	}
	std::cout << "GradientFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(g_h_gradientFeatures);

	t0 = cciutils::ClockGetTime();
	float* g_h_cannyFeatures = nscale::gpu::ObjFeatures::cannyFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "CannyFeaturesTimeGPU = "<< t1-t0 <<std::endl;

	free(g_h_cannyFeatures);

	g_gray.release();
	g_labeledImage.release();
	nscale::gpu::cudaFreeCaller(g_bbox);

#endif
	bgr[0].release();
	bgr[1].release();
	bgr[2].release();





	free(bbox);
	if(cytoplasmBB != NULL){
		free(cytoplasmBB);
	}




	return 0;

}


