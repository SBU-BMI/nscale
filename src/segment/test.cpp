/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "cv.h"
#include "highgui.h"

using namespace cv;

int main (int argc, char **argv){
	const char* imagename = argc > 1 ? argv[1] : "lena.jpg";

	Mat img = imread(imagename);

	if (!img.data) return -1;

	namedWindow("original image", CV_WINDOW_AUTOSIZE);
	imshow("original image", img);

	waitKey();

	return 0;
}
