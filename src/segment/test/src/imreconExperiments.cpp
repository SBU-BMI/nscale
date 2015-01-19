/*
 *
 *  Created on: Jan 13, 2012
 *      Author: george
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"
#include <stdio.h>


#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "opencv2/gpu/gpu.hpp"



using namespace cv;
using namespace cv::gpu;
using namespace std;

int main (int argc, char **argv){
	if(argc!= 6){
		printf("Usage: ./imreconTest <numImages> <numFirstPasses> <connectivity(4,8)> <marker> <mask>");
		exit(1);
	}
	gpu::setDevice(0);
	// Used to get store timestamp and calc. exec. times
	uint64_t t1, t2;

	// OpenCV Stream to used by our functions
	Stream stream;

	// Objects to store the maker and mask images in the GPU
	GpuMat g_marker, g_mask;

	// Store the reconstructed image in the GPU
	GpuMat g_recon;

	// Store the reconstructed image for the CPU case
	Mat recon;

	// "Get" input parameters
	int num_images = atoi(argv[1]);
	int numFirstPasses = atoi(argv[2]);
	int connectivity=atoi(argv[3]);
	
	// Read marker and mask images	
	Mat marker = imread(argv[4], -1);
	Mat mask = imread(argv[5], -1);

	// Asser the reading operations worked fine
	assert(marker.cols != 0  && marker.rows != 0 && mask.cols == marker.cols && mask.rows == marker.rows);

	// When this value is higher than 1, the input image is "replicated" and the actually image use is 
	// increased (Zoom in) by the factor chosen 
	int zoomFactor = 1;
	cout << "Zoom = "<< zoomFactor<<endl;

	cout << "Marker.type = "<< marker.type() << " cols="<< marker.cols << " rows="<< marker.rows <<endl;

	if(zoomFactor > 1){
		Mat bigMarker( marker.rows * zoomFactor, marker.cols * zoomFactor, marker.type());
		for(int x = 0; x < zoomFactor; x++){
			for(int y = 0; y < zoomFactor; y++){
				Mat roi = cv::Mat(bigMarker, cv::Rect(x*marker.cols, y*marker.rows,marker.cols, marker.rows));
				marker.copyTo(roi);
			}
		}

		Mat bigMask( marker.rows * zoomFactor, marker.cols * zoomFactor, marker.type());
		for(int x = 0; x < zoomFactor; x++){
			for(int y = 0; y < zoomFactor; y++){
				Mat roi = cv::Mat(bigMask, cv::Rect(x*marker.cols, y*marker.rows,marker.cols, marker.rows));
				mask.copyTo(roi);
			}
		}

		marker.release();
		marker = bigMarker;

		mask.release();
		mask = bigMask;
	}


	stream.enqueueUpload(marker, g_marker);
	stream.enqueueUpload(mask, g_mask);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;
////	//	int connectivity = 4;
		for(int numPasses=1; numPasses < numFirstPasses; numPasses+=1){
			Mat recon2;
			// 4 connectivity
			t1 = cci::common::event::timestampInUS();
			g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_marker, g_mask, connectivity, numPasses,stream, 32, false);
//			g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_marker, g_mask, connectivity, numPasses,stream);
			stream.waitForCompletion();
			t2 = cci::common::event::timestampInUS();
			g_recon.download(recon2);
			imwrite("test/out-gpu-queueu.ppm", recon2);
			recon2.release();
			cout << "gpu queue_recon"<< connectivity<< " passes "<< numPasses <<" took " << t2-t1<< " ms"<<endl;
			cout << "gpu queue_recon"<< connectivity<< " passes "<< numFirstPasses<<" took " << t2-t1<< " ms"<<endl;
			g_recon.release();
		}
//		int maxBlocks=48;
//		for(int numBlocks=1; numBlocks < maxBlocks; numBlocks+=1){
//			// 4 connectivity
//			t1 = cci::common::event::timestampInUS();
//
//			g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_marker, g_mask, connectivity, numFirstPasses,stream,numBlocks);
//			stream.waitForCompletion();
//			t2 = cci::common::event::timestampInUS();
//			cout << "gpu queue_recon"<< connectivity<< " nBlocks "<< numBlocks<<" took " << t2-t1<< " ms"<<endl;
//			g_recon.release();
//		}
//
//

//		cout << "Connectivity="<<connectivity<<endl;
//		t1 = cci::common::event::timestampInUS();
//		g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, connectivity, stream);
//		stream.waitForCompletion();
//		t2 = cci::common::event::timestampInUS();
//		std::cout << "gpu recon"<< connectivity <<" took " << t2-t1 << " ms" << std::endl;
//		g_recon.release();

//		t1 = cci::common::event::timestampInUS();
//		recon = nscale::imreconstruct<unsigned char>(marker, mask, connectivity);
//		t2 = cci::common::event::timestampInUS();
//		std::cout << "recon"<< connectivity <<" took " << t2-t1 << "ms" << std::endl;
//		imwrite("test/out-recon8.ppm", recon);
//		recon.release();


	g_marker.release();
	g_mask.release();
	marker.release();
	mask.release();

	return 0;
}

