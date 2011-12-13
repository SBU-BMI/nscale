/*
 * test.cpp
 *
 *  Created on: Dez 12, 2011
 *      Author: george
 */
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <errno.h>
#include <time.h>
#include "PixelOperations.h"
#include "MorphologicOperations.h"
#include "utils.h"
#include <stdio.h>

#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

int main(int argc, char** argv) {
	//initialize stain deconvolution matrix and channel selection matrix
	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

	//read image
	Mat image;
	image = imread( argv[1], 1 );  //For each pixel, BGR
	if( argc != 2 || !image.data )
	{
		printf( "No image data \n" );
		return -1;
	}

	//specify if color channels should be re-ordered
	bool BGR2RGB = true;

	//initialize H and E channels
	Mat H = Mat::zeros(image.size(), CV_8UC1);
	Mat E = Mat::zeros(image.size(), CV_8UC1);

	long t1 = cciutils::ClockGetTime();
	//color deconvolution
	nscale::PixelOperations::ColorDeconv( image, M, b, H, E, BGR2RGB);

	long t2 = cciutils::ClockGetTime();
	cout << "Conv original = "<< t2-t1<<endl;

	Stream stream;
	GpuMat g_image = GpuMat(image.size(), image.type());
	GpuMat g_H = GpuMat(image.size(), CV_8UC1);
	GpuMat g_E = GpuMat(image.size(), CV_8UC1);
	// These lines are used for debugging purpose only
	::gpu::min(g_H, 0, g_H);
	::gpu::min(g_E, 0, g_E);

	stream.enqueueUpload(image, g_image);
	stream.waitForCompletion();
	g_image.release();

	stream.enqueueUpload(image, g_image);
	stream.waitForCompletion();

	long t1_gpu = cciutils::ClockGetTime();

	nscale::gpu::PixelOperations::ColorDeconv( g_image, M, b, g_H, g_E, stream, BGR2RGB);

	long t2_gpu = cciutils::ClockGetTime();

	cout << "Conv gpu = " << t2_gpu - t1_gpu <<endl;

	Mat c_H(g_H);
	Mat c_E(g_E);

	Mat gray = nscale::PixelOperations::bgr2gray(image);


	GpuMat g_gray = nscale::gpu::PixelOperations::bgr2gray(g_image, stream);

	Mat gray_gpu(g_gray);

	if(countNonZero(gray != gray_gpu) ){
		printf("Error: grayscale from CPU and GPU differ by %d pixels!\n", countNonZero(gray != gray_gpu));
		exit(1);
	}else{
		cout <<"Success: Grayscale Images computed by CPU and GPU are the same!" <<endl;
	}

//	imwrite("image_gray.ppm", gray);
//	cout << "Writing images"<<endl;
//	imwrite("gpu_E.ppm", c_E);
//	imwrite("E.ppm", E);
//
//	imwrite("gpu_H.ppm", c_H);
//	imwrite("H.ppm", H);

	if(countNonZero(H != c_H) || countNonZero(E != c_E)){
		printf("Error: E or H images are not the same! H = %d and E = %d\n", countNonZero(H != c_H), countNonZero(E != c_E));
		exit(1);
	}else{
		cout <<"Success: Images computed by CPU and GPU are the same!" <<endl;
	}
	// release GPU memory
	g_image.release();

	//release CPU memory
	H.release();
	E.release();
	M.release();
	b.release();
	image.release();

    return 0;
}

//int main (int argc, char **argv){
//	Mat marker = imread("test/in-imrecon-gray-marker.pgm", -1);
//	Mat mask = imread("test/in-imrecon-gray-mask.pgm", -1);
//	Mat markerb = marker > 64;
//	Mat maskb = mask > 32;
//	
//	Mat recon, recon2;
//	uint64_t t1, t2;
//
//	Stream stream;
//	GpuMat g_marker, g_marker1;
//	GpuMat g_mask, g_mask1, g_recon;
//
//	stream.enqueueUpload(marker, g_marker);
//	stream.enqueueUpload(mask, g_mask);
//	stream.enqueueUpload(marker, g_marker1);
//	stream.enqueueUpload(mask, g_mask1);
//
//
//	stream.waitForCompletion();
//	std::cout << "finished uploading" << std::endl;
//
//
//	return 0;
//}

