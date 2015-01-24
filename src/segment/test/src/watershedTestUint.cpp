#include "opencv/cv.hpp"


#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::gpu;
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
#include "PixelOperations.h"
#include "NeighborOperations.h"

#include "Logger.h"
#include <stdio.h>


#if defined (WITH_CUDA)
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/stream_accessor.hpp"
#endif

using namespace cv;
using namespace cv::gpu;


int main (int argc, char **argv){

	if(argc != 3){
		printf("Usage: ./imreconTest <inputImage> <connectivity=4|8>");
	}
	Mat input = imread(argv[1], -1);
	int connectivity = atoi(argv[2]);
	std::cout << "Cols: " << input.cols << " Rows: "<< input.rows<< std::endl; 
	if(input.channels() == 3){
		input = nscale::PixelOperations::bgr2gray(input);

		imwrite("in-cpu-watershed-gray.png", input);
	}
	Mat auxInput;
	input.convertTo(auxInput, CV_16U);
	input = auxInput;


	uint64_t t1, t2;

	std::cout << "Cols: " << input.cols << " Rows: "<< input.rows<< std::endl; 
	t1 = cci::common::event::timestampInUS();

	//imwrite("in-cpu-watershed.png", input);
	Mat waterResult;
	waterResult = nscale::watershed(input, connectivity);

	t2 = cci::common::event::timestampInUS();
	std::cout << "cpu watershed loop took " << (t2-t1)/1000 << "ms" << std::endl;

	imwrite("out-watershed.ppm", waterResult);

// Debuging data, same as used in the paper: An efficient watershed algorithm based on connected components
/*	unsigned short int testIn[11][10] = {
			{3,5,5,2,8,8,8,11,10,10},
			{5,5,11,11,8,11,11,8,10,10},
			{11,5,11,11,9,9,9,9,8,10},
			{11,11,11,7,7,7,7,9,9,8},
			{11,11,11,11,11,9,7,10,8,10},
			{11,10,11,9,7,7,9,9,10,8},
			{11,10,11,9,11,9,10,10,8,10},
			{11,11,11,8,8,8,8,8,10,10},
			{11,11,11,11,10,10,10,10,10,10},
			{10,10,10,10,10,10,10,10,10,10},
			{11,11,11,11,10,10,10,10,10,10}
	};
	cv::Mat testInMat = Mat(11,10, CV_16U, &testIn);

	Mat waterResultCCIn = nscale::watershedCC(testInMat, connectivity);
	imwrite("out-watershedCC-test.ppm", waterResultCCIn);

	exit(1);*/
	t1 = cci::common::event::timestampInUS();

	//imwrite("in-cpu-watershed.png", input);

	Mat waterResultCC = nscale::watershedCC(input, connectivity);

	t2 = cci::common::event::timestampInUS();
	std::cout << "cpu watershedCC loop took " << (t2-t1)/1000 << "ms" << std::endl;

	imwrite("out-watershedCC.ppm", waterResultCC);
	return 0;
}

