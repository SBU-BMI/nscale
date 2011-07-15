/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <errno.h>
#include "RedBloodCell.h"
#include <time.h>
#include "MorphologicOperation.h"
#include "utils.h"

using namespace cv;


int main (int argc, char **argv){
/*	// allow walk through of the directory
	const char* impath = argc > 1 ? argv[1];
	// get the files - from http://ubuntuforums.org/showthread.php?t=1409202
	vector<string> files();
	Dir *dir;
	struct dirent *dp;
	if ((dir = std::opendir(impath.c_str())) == NULL) {
		std::cout << "ERROR(" << errno << ") opening" << impath << std::endl;
		return errno;
	}
	while ((dp = readdir(dir)) != NULL) {
		files.push_back(string(dp->d_name));
		if ()
	}
	closedir(dir);


	// set the output path
	const char* resultpath = argc > 2 ? argv[2];
*/
	const char* imagename = argc > 1 ? argv[1] : "lena.jpg";

	// need to go through filesystem

	Mat img = imread(imagename);

	if (!img.data) return -1;


	// image in BGR format

//	Mat img2(Size(1024,1024), img.type());
//	resize(img, img2, Size(1024,1024));
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", img2);

	/*
	* this part to decide if the tile is background or foreground
	THR = 0.9;
    grayI = rgb2gray(I);
    area_bg = length(find(I(:,:,1)>=220&I(:,:,2)>=220&I(:,:,3)>=220));
    ratio = area_bg/numel(grayI);
    if ratio >= THR
        return;
    end
	 */
	Mat gray(img.size(), CV_8UC1);
	cvtColor(img, gray, CV_BGR2GRAY);

	std::vector<Mat> bgr;
	split(img, bgr);
	Mat background = (bgr[0] > 220) & (bgr[1] > 220) & (bgr[2] > 220);
	int bgArea = countNonZero(background);
	float ratio = (float)bgArea / (float)(img.size().area());
	if (ratio >= 0.9) {
		std::cout << "background.  next." << std::endl;
		return -1;
	}

	uint64_t t1 = cciutils::ClockGetTime();

	Mat rbc = nscale::RedBloodCell::rbcMask(bgr);

	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "rbc took " << t2-t1 << "ms" << std::endl;

	imwrite("/home/tcpan/PhD/path/rbc.pbm", rbc);

//	resize(rbc, img2, Size(1024,1024));
//	namedWindow("rbc image", CV_WINDOW_AUTOSIZE);
//	imshow("rbc image", img2);


	Mat rc = 255 - bgr[2];
	Mat rc_dilate(rc.size(), rc.type());

//	waitKey();

	return 0;
}
