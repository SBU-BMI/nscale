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

	cv::Mat img = cv::imread(imagename);

	if (!img.data) return -1;
	cv::namedWindow("orig image", CV_WINDOW_AUTOSIZE);
	cv::imshow("orig image", img);


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

	cv::Mat gray(img.size(), CV_8UC1);
	cv::cvtColor(img, gray, CV_RGB2GRAY);

	std::vector<cv::Mat> rgb;
	cv::split(img, rgb);
	cv::Mat background(img.size(), CV_8UC1);
	background = (rgb[0] > 220) & (rgb[1] > 220) & (rgb[2] > 220);
	int bgArea = cv::countNonZero(background);
	float ratio = (float)bgArea / (float)(img.size().area());
	if (ratio >= 0.9) {
		std::cout << "background.  next." << std::endl;
		return -1;
	}





	cv::waitKey();

	return 0;
}
