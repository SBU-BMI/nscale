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
	uint64_t t1, t0, tinit, tend;


	nscale::HistologicalEntities::segmentNuclei(inputImg, labeledImage, compCount, bbox);
//	int* objsArea = nscale::ObjFeatures::area((const int *)bbox, compCount, labeledImage);
//	free(objsArea);
	int cytoplasmDataSize;
	t0 = cciutils::ClockGetTime();
	tinit = t0;
	int* cytoplasmBB = nscale::CytoplasmCalc::calcCytoplasm(cytoplasmDataSize, bbox, compCount, labeledImage);
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
	for(int i = 0; i < 24; i++){
		cout << "cytoIntensityCPU id["<<i<<"] = "<< cytoIntensityFeatures[i]<<endl;
	}
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

	int*g_cyto_bbox = (int*)nscale::gpu::cudaMallocCaller(cytoplasmDataSize);
	nscale::gpu::cudaUploadCaller(g_cyto_bbox, cytoplasmBB, cytoplasmDataSize);

	stream.enqueueUpload(labeledImage, g_labeledImage);
	stream.enqueueUpload(bgr[0], g_gray);
	stream.waitForCompletion();


	t0 = cciutils::ClockGetTime();
	float* h_intensityFeatures = nscale::gpu::ObjFeatures::intensityFeatures(g_bbox,  compCount, g_labeledImage , g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "NucleiIntensityFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(h_intensityFeatures);

	t0 = cciutils::ClockGetTime();
	float* h_cytoIntensityFeatures = nscale::gpu::ObjFeatures::cytoIntensityFeatures(g_cyto_bbox,  compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "CytoIntensityFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	for(int i = 0; i < 24; i++){
		cout << "cytoIntensityGPU id["<<i<<"] = "<< h_cytoIntensityFeatures[i]<<endl;
	}
	free(h_cytoGradientFeatures);


	t0 = cciutils::ClockGetTime();
	float* g_h_gradientFeatures = nscale::gpu::ObjFeatures::gradientFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
//	for(int i = 0; i < 12; i++){
//		cout << "GradFeatureGPU id["<<i<<"] = "<< g_h_gradientFeatures[i]<<endl;
//	}
	std::cout << "NucleiGradientFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(g_h_gradientFeatures);

	t0 = cciutils::ClockGetTime();
	float* g_h_cytoGradientFeatures = nscale::gpu::ObjFeatures::cytoGradientFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
//	for(int i = 0; i < 12; i++){
//		cout << "GradFeatureGPU id["<<i<<"] = "<< g_h_gradientFeatures[i]<<endl;
//	}
	std::cout << "CytoGradientFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(g_h_cytoGradientFeatures);

	t0 = cciutils::ClockGetTime();
	float* g_h_cannyFeatures = nscale::gpu::ObjFeatures::cannyFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "NucleiCannyFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(g_h_cannyFeatures);

	t0 = cciutils::ClockGetTime();
	float* g_h_cytoCannyFeatures = nscale::gpu::ObjFeatures::cytoCannyFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cciutils::ClockGetTime();
	std::cout << "CytoCannyFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	free(g_h_cytoCannyFeatures);

	std::cout << "FeaturesTime="<< t1-tinit<<std::endl;
	g_gray.release();
	g_labeledImage.release();
	nscale::gpu::cudaFreeCaller(g_bbox);
	nscale::gpu::cudaFreeCaller(g_cyto_bbox);
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


