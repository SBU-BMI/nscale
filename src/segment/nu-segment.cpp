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
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include <time.h>
#include "utils.h"


using namespace cv;

bool areaThreshold1(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx) {
	return nscale::contourAreaFilter(contours, hierarchy, idx, 11, 1000);
}


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

	Mat output = Mat::zeros(img.size(), CV_8U);
	int status = nscale::HistologicalEntities::segmentNuclei(img, output);



//	waitKey();

	return status;
}


