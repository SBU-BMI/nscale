/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "utils.h"

#include <iostream>
#include <iomanip>


using namespace cv;


int main (int argc, char **argv){

	if(argc != 2){
		std::cout << "./distTransform <maskImage>" << std::endl;
		exit(1);
	}
	Mat input = imread(argv[1], -1);
	if(input.data == NULL){
		printf("Failed reading");
		exit(1);
	}
	std::cout << "input - " << (int) input.ptr(10)[20] << std::endl;
gpu::setDevice(0);
//	Mat point(4000,4000, CV_8UC1);
//	point.ones(4000,4000, CV_8UC1);
//
//	for(int x = 0; x < point.rows; x++){
//		uchar* ptr = point.ptr(x);
//		for(int y = 0; y < point.cols; y++){
//			ptr[y] = 1;
//			if(x==1 && y==3 || x==4&&y==0 || x==3&&y==0 || x==3&&y==1 || x==4&&y==1){
//				ptr[y] = 0;
//			}
////			std::cout << (int) ptr[y] <<" ";
//		}
////		std::cout<<std::endl;
//	}
//	uchar *ptr = point.ptr(1);
//	ptr[3] = 0;

	Mat dist(input.size(), CV_32FC1);


	uint64_t t1 = cciutils::ClockGetTime();
	distanceTransform(input, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	uint64_t t2 = cciutils::ClockGetTime();

	std::cout << "distTransf CPU  took " << t2-t1 <<" ms"<<std::endl;
//	
//	for(int x = 0; x < dist.rows; x++){
//		float* ptr = (float*)dist.ptr(x);
//		for(int y = 0; y < dist.cols; y++){
//			std::cout << std::setprecision(10) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}

	GpuMat g_mask(input);
	
	Stream stream;

	t1 = cciutils::ClockGetTime();
	GpuMat g_distance = nscale::gpu::distanceTransform(g_mask, stream);

	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "distTransf GPU  took " << t2-t1 <<" ms"<<std::endl;

	Mat h_distance(g_distance);
//	for(int x = 0; x < h_distance.rows; x++){
//		float* ptr = (float*)h_distance.ptr(x);
//		for(int y = 0; y < h_distance.cols; y++){
//			std::cout << std::setprecision(10) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}

	Mat diff = (h_distance - dist) >0.01;
	std::cout << "NonZero=" << countNonZero(diff) << std::endl;
	h_distance.release();
	g_distance.release();
	g_mask.release();


	return 0;
}

