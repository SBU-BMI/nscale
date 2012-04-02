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
#include "PixelOperations.h"
#include "utils.h"

#include <iostream>
#include <iomanip>


using namespace cv;


int main (int argc, char **argv){

	if(argc != 1){
		std::cout << "./copyMakeBorder" << std::endl;
		exit(1);
	}
//	gpu::setDevice(0);

	Mat src(4096, 4096, CV_32FC1);
	randn(src, Scalar::all(100.5), Scalar::all(10.1) );

	Mat dst(src.rows+2, src.cols+2, CV_32FC1);
	copyMakeBorder(src, dst, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(1.1));

	Stream stream;
	GpuMat g_src(src);
	GpuMat g_dst(dst.rows, dst. cols, CV_32FC1);

	nscale::gpu::PixelOperations::copyMakeBorder(g_src, g_dst, 1, 1, 1, 1, Scalar(1.1), stream);
	stream.waitForCompletion();

	Mat cpu_dst(g_dst);
	std::cout << countNonZero(dst-cpu_dst) <<std::endl;
	

//	uint64_t t1 = cciutils::ClockGetTime();
//	distanceTransform(input, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
//	uint64_t t2 = cciutils::ClockGetTime();
//
//	std::cout << "distTransf CPU  took " << t2-t1 <<" ms"<<std::endl;
////	
////	for(int x = 0; x < dist.rows; x++){
////		float* ptr = (float*)dist.ptr(x);
////		for(int y = 0; y < dist.cols; y++){
////			std::cout << std::setprecision(10) << ptr[y] <<"\t ";
////		}
////		std::cout<<std::endl;
////	}
//
//	GpuMat g_mask(input);
//	
//	Stream stream;
//
//	t1 = cciutils::ClockGetTime();
//	GpuMat g_distance = nscale::gpu::distanceTransform(g_mask, stream);
//
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "distTransf GPU  took " << t2-t1 <<" ms"<<std::endl;
//
//	Mat h_distance(g_distance);
////	for(int x = 0; x < h_distance.rows; x++){
////		float* ptr = (float*)h_distance.ptr(x);
////		for(int y = 0; y < h_distance.cols; y++){
////			std::cout << std::setprecision(10) << ptr[y] <<"\t ";
////		}
////		std::cout<<std::endl;
////	}
//
//	Mat diff = (h_distance - dist) >0.01;
//	std::cout << "NonZero=" << countNonZero(diff) << std::endl;
//	h_distance.release();
//	g_distance.release();
//	g_mask.release();


	return 0;
}

