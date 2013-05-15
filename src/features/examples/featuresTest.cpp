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
#include "Logger.h"

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
	t0 = cci::common::event::timestampInUS();
	tinit = t0;
	int* cytoplasmBB = nscale::CytoplasmCalc::calcCytoplasm(cytoplasmDataSize, bbox, compCount, labeledImage);
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoplasmMaskCalc = "<< t1-t0 <<std::endl;

	vector<cv::Mat> bgr;
	split(inputImg, bgr);
	t0 = cci::common::event::timestampInUS();
	float* h_intensityFeatures = nscale::ObjFeatures::intensityFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiIntensityFeaturesTime = "<< t1-t0 <<std::endl;

	t0 = cci::common::event::timestampInUS();
	float* h_cytoIntensityFeatures = nscale::ObjFeatures::cytoIntensityFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoIntensityFeaturesTime = "<< t1-t0 <<std::endl;
//	for(int i = 0; i < 24; i++){
//		cout << "cytoIntensityCPU id["<<i<<"] = "<< cytoIntensityFeatures[i]<<endl;
//	}

	t0 = cci::common::event::timestampInUS();
	float* h_gradientFeatures = nscale::ObjFeatures::gradientFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiGradientFeaturesTimeCPU = "<< t1-t0 <<std::endl;


	t0 = cci::common::event::timestampInUS();
	float* h_cytoGradientFeatures = nscale::ObjFeatures::cytoGradientFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoGradientFeaturesTimeCPU = "<< t1-t0 <<std::endl;


	t0 = cci::common::event::timestampInUS();
	float* h_cannyFeatures = nscale::ObjFeatures::cannyFeatures(bbox, compCount, labeledImage, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiCannyFeaturesTimeCPU = "<< t1-t0 <<std::endl;

	t0 = cci::common::event::timestampInUS();
	float* h_cytoCannyFeatures = nscale::ObjFeatures::cytoCannyFeatures(cytoplasmBB, compCount, bgr[0]);
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoCannyFeaturesTimeCPU = "<< t1-t0 <<std::endl;

#if defined(WITH_CUDA)
	Stream stream;
	GpuMat g_labeledImage;
	GpuMat g_gray;
	double err;
	double terr;

	int*g_bbox = (int*)nscale::gpu::cudaMallocCaller(sizeof(int) * 5 * compCount);
	nscale::gpu::cudaUploadCaller(g_bbox, bbox, sizeof(int) * 5 * compCount);

	int*g_cyto_bbox = (int*)nscale::gpu::cudaMallocCaller(cytoplasmDataSize);
	nscale::gpu::cudaUploadCaller(g_cyto_bbox, cytoplasmBB, cytoplasmDataSize);

	stream.enqueueUpload(labeledImage, g_labeledImage);
	stream.enqueueUpload(bgr[0], g_gray);
	stream.waitForCompletion();


	t0 = cci::common::event::timestampInUS();
	float* g_h_intensityFeatures = nscale::gpu::ObjFeatures::intensityFeatures(g_bbox,  compCount, g_labeledImage , g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiIntensityFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_intensityFeatures[i] - h_intensityFeatures[i];
		err += terr * terr;
	}
	std::cout << "intensityFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_intensityFeatures);

	t0 = cci::common::event::timestampInUS();
	float* g_h_cytoIntensityFeatures = nscale::gpu::ObjFeatures::cytoIntensityFeatures(g_cyto_bbox,  compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoIntensityFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_cytoIntensityFeatures[i] - h_cytoIntensityFeatures[i];
		err += terr * terr;
	}
	std::cout << "CytontensityFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_cytoIntensityFeatures);


	t0 = cci::common::event::timestampInUS();
	float* g_h_gradientFeatures = nscale::gpu::ObjFeatures::gradientFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiGradientFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_gradientFeatures[i] - h_gradientFeatures[i];
		err += terr * terr;
	}
	std::cout << "gradientFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_gradientFeatures);

	t0 = cci::common::event::timestampInUS();
	float* g_h_cytoGradientFeatures = nscale::gpu::ObjFeatures::cytoGradientFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoGradientFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_cytoGradientFeatures[i] - h_cytoGradientFeatures[i];
		err += terr * terr;
	}
	std::cout << "CytoGradientFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_cytoGradientFeatures);

	t0 = cci::common::event::timestampInUS();
	float* g_h_cannyFeatures = nscale::gpu::ObjFeatures::cannyFeatures(g_bbox, compCount, g_labeledImage, g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "NucleiCannyFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_cannyFeatures[i] - h_cannyFeatures[i];
		err += terr * terr;
	}
	std::cout << "cannyFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_cannyFeatures);

	t0 = cci::common::event::timestampInUS();
	float* g_h_cytoCannyFeatures = nscale::gpu::ObjFeatures::cytoCannyFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();
	t1 = cci::common::event::timestampInUS();
	std::cout << "CytoCannyFeaturesTimeGPU = "<< t1-t0 <<std::endl;
	err = 0;
	for(int i = 0; i < compCount; i++){
		terr = g_h_cytoCannyFeatures[i] - h_cytoCannyFeatures[i];
		err += terr * terr;
	}
	std::cout << "cytoCannyFeatures CPU vs GPU rmse " << sqrt(err / compCount) << std::endl;
	free(g_h_cytoCannyFeatures);

	std::cout << "FeaturesTime="<< t1-tinit<<std::endl;
	g_gray.release();
	g_labeledImage.release();
	nscale::gpu::cudaFreeCaller(g_bbox);
	nscale::gpu::cudaFreeCaller(g_cyto_bbox);
#endif
/*	bgr[0].release();
	bgr[1].release();
	bgr[2].release();
*/



	free(h_intensityFeatures);
	free(h_cytoIntensityFeatures);
	free(h_gradientFeatures);
	free(h_cytoGradientFeatures);
	free(h_cannyFeatures);
	free(h_cytoCannyFeatures);

	free(bbox);
	if(cytoplasmBB != NULL){
		free(cytoplasmBB);
	}




	return 0;

}


