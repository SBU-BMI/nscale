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

	time_t first = time(NULL);

	Mat rbc = nscale::RedBloodCell::rbcMask(bgr);
	imwrite("/home/tcpan/PhD/path/rbc.pbm", rbc);

//	resize(rbc, img2, Size(1024,1024));
//	namedWindow("rbc image", CV_WINDOW_AUTOSIZE);
//	imshow("rbc image", img2);

	Mat rbc2(rbc.size(), rbc.type());
	Mat el = getStructuringElement(MORPH_RECT, Size(3,3));
	dilate(rbc, rbc2, el, Point(-1, -1), 3);
	imwrite("/home/tcpan/PhD/path/rbc2.pbm", rbc2);


	Mat_<uchar> rbc2_ = rbc2;
	Mat_<uchar> rbc_ = rbc;
	Mat_<uchar> out = nscale::imreconstruct(rbc2_, rbc_, 8);
	imwrite("/home/tcpan/PhD/path/imrecon.pbm", out);



	Mat rc = 255 - bgr[2];
	Mat rc_dilate(rc.size(), rc.type());


	time_t second = time(NULL);
	double elapsed = difftime(second, first);
	std::cout << "rbc took " << elapsed << " s" << std::endl;

//	waitKey();

	return 0;
}
