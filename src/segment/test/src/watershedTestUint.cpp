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
#include <string>
#include <fstream>
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


// http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv 
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

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

	imwrite(string(argv[1]) + "in_watershed.png", input);

	std::cout << "Cols: " << input.cols << " Rows: "<< input.rows<< std::endl; 
	t1 = cci::common::event::timestampInUS();

	//imwrite("in-cpu-watershed.png", input);
	Mat waterResult;
	waterResult = nscale::watershed(input, connectivity);

	t2 = cci::common::event::timestampInUS();

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
	//watershed seems to return int32_t matrix
	//check http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv to interpret type
	std::cout << "cpu watershed loop took " << (t2-t1)/1000 << "ms. Output type is "<<type2str( waterResult.type() )<< ". Depth = "<<waterResult.depth()<<". Channels = "<<waterResult.channels()<< std::endl;

	std::string fileout(argv[1]);
	fileout.append("out_watershed.ppm");
	imwrite(fileout.c_str(), waterResult);

	std::string fileout2(argv[1]);
	fileout2.append("out_watershed.raw");
	std::ofstream fout(fileout2.c_str(), std::ios::binary);

	vector<int> img;
	img.reserve(input.cols*input.rows);
	for (int i = 0; i < waterResult.rows; ++i)
		for (int j = 0; j < waterResult.cols; ++j)
			img.push_back(waterResult.at<int>(i, j));
	//fout.write((char*)(waterResult.data), sizeof(int) * waterResult.cols * waterResult.rows);
	fout.write((char*)(&(img[0])), sizeof(int)* img.size());
	fout.close();

	imwrite(string(argv[1]) + "out_watershed.png", waterResult);	
	

	t1 = cci::common::event::timestampInUS();

	//imwrite("in-cpu-watershed.png", input);

	Mat waterResultCC = nscale::watershedCC(input, connectivity);

	t2 = cci::common::event::timestampInUS();

	std::cout << "cpu watershedCC loop took " << (t2-t1)/1000 << " ms" << std::endl;
	
	fileout = string(argv[1]);
	fileout.append("out_watershedCC.ppm");
	imwrite(fileout.c_str(), waterResultCC);

	fileout2 = string(argv[1]);
	fileout2.append("out_watershedCC.raw");
	fout.open(fileout2.c_str(), std::ios::binary);

	img.clear();
	for (int i = 0; i < waterResultCC.rows; ++i)
		for (int j = 0; j < waterResultCC.cols; ++j)
			img.push_back(waterResultCC.at<int>(i, j));
	//fout.write((char*)(waterResultCC.data), sizeof(int)* waterResult.cols * waterResult.rows);
	fout.write((char*)(&(img[0])), sizeof(int)* img.size());
	fout.close();

	imwrite(string(argv[1]) + "out_watershedCC.png", waterResultCC);

	return 0;
}

