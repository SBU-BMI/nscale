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
#include "Logger.h"


using namespace cv;


int main (int argc, char **argv){

	Mat img1 = imread(argv[1], -1);
	Mat img2 = imread(argv[2], -1);
	std::cout << "img1.nChannels="<< img1.channels() << std::endl;
	std::cout << "imt2.nChannels="<< img2.channels() << std::endl;

	Mat img1mask = img1 > 0;	
	Mat img2mask = img2 > 0;

	
	Mat common = (img1mask+img2mask) > 1;

	Mat firstNotSecond = (img1mask-img2mask) > 0;
	std::cout << "Non-zero = "<< countNonZero(firstNotSecond) << std::endl;

	// In second, and not first
	Mat secondNotFirst = (img2mask-img1mask) > 0;

	Mat colorDiff (img1.size(), CV_8UC3);


	colorDiff.setTo(Scalar(0,255,0), common);
	colorDiff.setTo(Scalar(255, 0, 0), firstNotSecond);
	colorDiff.setTo(Scalar(0, 0, 255), secondNotFirst);


//	namedWindow("diff", CV_WINDOW_AUTOSIZE);
//	imshow("diff", colorDiff);
	imwrite("diff.pbm", colorDiff);

//	imwrite("test/out-gray.tif", gray);
//	imwrite("test/out-gray.ppm", gray);

	waitKey();

	return 0;
}

