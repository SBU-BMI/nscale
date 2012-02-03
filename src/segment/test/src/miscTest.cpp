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
#include <string>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "utils.h"


using namespace cv;


int main (int argc, char **argv){

	const char* imagename = argc > 1 ? argv[1] : "lena.jpg";

	// need to go through filesystem

	Mat img = imread(imagename);

	if (!img.data) return -1;
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", img);



	Mat gray(img.size(), CV_8UC1);
	cvtColor(img, gray, CV_BGR2GRAY);
//	namedWindow("gray image", CV_WINDOW_AUTOSIZE);
//	imshow("gray image", gray);
	imwrite("test/out-gray.tif", gray);
	imwrite("test/out-gray.ppm", gray);

//	waitKey();

	return 0;
}

