/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "highgui.h"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
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
	if (argc < 2) {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "[cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	const char* imagename = argv[1];
	const char* mode = argc > 2 ? argv[2] : "cpu";

	int modecode = 0;
	if (strcasecmp(mode, "cpu") == 0) modecode = cciutils::DEVICE_CPU;
	else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			std::cout << "gpu requested, but no gpu available.  please use cpu or mcore option."  << std::endl;
			return -2;
		}
		if (argc > 3) {
			gpu::setDevice(atoi(argv[3]));
		}
		std::cout << " number of cuda enabled devices = " << gpu::getCudaEnabledDeviceCount() << std::endl;
	} else {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "[cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}

	// need to go through filesystem

	Mat img = imread(imagename);

	if (!img.data) return -1;

	uint64_t t1 = cciutils::ClockGetTime();
	Mat output = Mat::zeros(img.size(), CV_8U);
	int status;
	switch (modecode) {
	case cciutils::DEVICE_CPU :
	case cciutils::DEVICE_MCORE :
		status = nscale::HistologicalEntities::segmentNuclei(img, output);
		break;
	case cciutils::DEVICE_GPU :
		status = nscale::gpu::HistologicalEntities::segmentNuclei(img, output);
		break;
	default :
		break;
	}
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "segment took " << t2-t1 << "ms" << std::endl;



	waitKey();

	return status;
}


