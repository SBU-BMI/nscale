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
	if (argc < 3) {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id width height [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	const char* imagename = argv[1];
	const char* runid = argv[2];
	int w = atoi(argv[3]);
	int h = atoi(argv[4]);
	const char* mode = argc > 5 ? argv[5] : "cpu";

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
		if (argc > 4) {
			gpu::setDevice(atoi(argv[6]));
		}
		std::cout << " number of cuda enabled devices = " << gpu::getCudaEnabledDeviceCount() << std::endl;
	} else {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id  [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}

	// need to go through filesystem
	Mat img = imread(imagename);

	// break it apart
	std::vector<Mat> chunks;
	Size s = img.size();
	int colw = s.width/w + (s.width%w == 0 ? 0 : 1);
	int rowh = s.height/h + (s.height%w == 0 ? 0 : 1);
	Mat chunk;
	Rect roi;
	for (int i = 0; i < s.width; i+= w) {
		for (int j = 0; j < s.height; j+= h) {
			roi = Rect(i, j, w, h);
			chunks.push_back(Mat(img, roi));
		}
	}
	
	if (!img.data) return -1;

	uint64_t t1 = cciutils::ClockGetTime();
	int status;
	char prefix[80];
	strcpy(prefix, "results/");
	strcat(prefix, mode);
	strcat(prefix, "-chunk");
	cciutils::SimpleCSVLogger logger(prefix);
	logger.log("run-id", runid);
	logger.log("w", w);
	logger.log("h", h);
	logger.log("time", cciutils::ClockGetTime());
	logger.log("filename", imagename);
	
	std::vector<Mat>::const_iterator it = chunks.begin();
	std::vector<Mat>::const_iterator last = chunks.end();

	std::vector<Mat> outputs;
	Mat output;
	Mat outputWhole;
	switch (modecode) {
	case cciutils::DEVICE_CPU :
	case cciutils::DEVICE_MCORE :
		logger.log("type", "cpu");
		for (; it < last; ++it) {
			status = nscale::HistologicalEntities::segmentNuclei(*it, output, logger);
			outputs.push_back(output);
		}
		status = nscale::HistologicalEntities::segmentNuclei(img, outputWhole, logger);
		logger.endSession();
		break;
	case cciutils::DEVICE_GPU :
		logger.log("type", "gpu");
		status = nscale::gpu::HistologicalEntities::segmentNuclei(img, output, logger);
		logger.endSession();
		break;
	default :
		break;
	}
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "segment took " << t2-t1 << "ms" << std::endl;

	// concat.
	std::vector<Mat> bgr;
	Mat red = Mat::zeros(colw * w, rowh * h, CV_8UC3);
	Mat green = Mat::zeros(colw * w, rowh * h, CV_8U);
	Mat blue = Mat::zeros(colw * w, rowh * h, CV_8U);
	bgr.push_back(blue);
	bgr.push_back(green);
	bgr.push_back(red);
	
	int channel = 1;

	it = chunks.begin();
	last = chunks.end();
	
	Mat target;
	for (int i = 0; i < s.width; i+=w) {
		for (int j = 0; j < s.height; j+=h) {
			target = bgr[channel];
			
			roi = Rect(i, j, w, h);
			target(roi) = (*it);

			channel = channel %2 + 1;
		}
		channel = channel %2 + 1;
	}
	merge(bgr, output);
	
	char ws[10];
	sprintf(ws, "%d", w);
	char hs[10];
	sprintf(hs, "%d", h);
	
	strcpy(prefix, "test/out-segment-");
	strcat(prefix, mode);
	strcat(prefix, "-chunk-");
	strcat(prefix, ws);
	strcat(prefix, "x");
	strcat(prefix, hs);
	strcat(prefix, ".pgm");
	imwrite(prefix, output);
	
	
	strcpy(prefix, "test/out-segment-");
	strcat(prefix, mode);
	strcat(prefix, ".pgm");
	imwrite(prefix, outputWhole);
	
	
//	waitKey();

	return status;
}


