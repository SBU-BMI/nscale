/*
 * test.cpp
 *
 * single threaded testing.
 *
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */


#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include "Normalization.h"
#include "Logger.h"
#include "FileUtils.h"
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include "UtilsLogger.h"
#include "UtilsCVImageIO.h"


#if defined(_WIN32) || defined(_WIN64)
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

using namespace cv;



int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir, bool &debug,
		::cciutils::SimpleCSVLogger *&logger);
void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output);
void compute(const char *input, const char *mask, const char *output, const int modecode, bool &debug,
		::cciutils::SimpleCSVLogger *logger);

int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir, bool &debug,
		::cciutils::SimpleCSVLogger *&logger) {
	if (argc < 5) {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id <time | debug> [cpu | gpu [id]]" << std::endl;
		return -1;
	}
	imageName.assign(argv[1]);
	outDir.assign(argv[2]);
	debug = false;
	printf("running with %s\n", argv[4]);

	if (strcasecmp(argv[4], "time") == 0) {
		std::stringstream ss;
		ss << outDir << "/" << argv[3];

		logger = new ::cciutils::SimpleCSVLogger(ss.str().c_str());
		logger->consoleOn();
		logger->on();

	} else if (strcasecmp(argv[4], "debug") == 0) {
		debug = true;
	}
	printf("debug is %s\n", (debug ? "true" : "false"));

	const char* mode = argc > 5 ? argv[5] : "cpu";

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cci::common::type::DEVICE_CPU;
		// get core count


	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cci::common::type::DEVICE_MCORE;
		// get core count


	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cci::common::type::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
		if (argc > 6) {
			gpu::setDevice(atoi(argv[6]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id <time | debug> [cpu | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output) {

	// check to see if it's a directory or a file
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));
	exts.push_back(std::string(".png"));


	cci::common::FileUtils futils(exts);
	futils.traverseDirectory(imageName, filenames, cci::common::FileUtils::getFILE(), true);
	std::string dirname;
	if (filenames.size() == 1) {
		dirname = imageName.substr(0, imageName.find_last_of("/\\"));
	} else {
		dirname = imageName;
	}


	std::string temp, tempdir;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
			// generate the output file name
		temp = futils.replaceExt(filenames[i], ".mask.png");
		temp = cci::common::FileUtils::replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		cci::common::FileUtils::mkdirs(tempdir);
		seg_output.push_back(temp);
		// generate the bounds output file name
		temp = futils.replaceExt(filenames[i], ".bounds.csv");
		temp = cci::common::FileUtils::replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		cci::common::FileUtils::mkdirs(tempdir);
		bounds_output.push_back(temp);
	}

}




void compute(const char *input, const char *mask, const char *output, const int modecode, bool &debug,
		::cciutils::SimpleCSVLogger *logger ) {
	// compute

    ::cciutils::cv::IntermediateResultHandler *iwrite = NULL;
    if (debug)	{
    	std::vector<int> stages;
    	for (int stage = 0; stage <= 200; ++stage) {
    		stages.push_back(stage);
    	}

    	cci::common::FileUtils fu(".mask.png");
    	std::string fmask(mask);

        std::string prefix = fu.replaceExt(fmask, ".mask.png", "");
        std::string suffix;
        suffix.assign(".mask.png");
    	iwrite = new ::cciutils::cv::IntermediateResultWriter(prefix, suffix, stages);

    	//printf("creating a debugger\n");
    }


    if (logger) {
		logger->log("filename", std::string(input));
		logger->log("type", modecode);
    }


	int *bbox = NULL;
	int compcount;

	printf("processing %s\n", input);
//	float meanT[3] = {-0.451225340366, -0.0219714958221, 0.0335194170475};
	float meanT[3] = {-0.632356, -0.0516004, 0.0376543};
//	float stdT[3] = {0.148816049099, 0.0257016178221, 0.00884802173823};
	float stdT[3] =  {0.26235, 0.0514831, 0.0114217};

	cv::Mat maskMat, inputImg, normalized;

	switch (modecode) {
	case cci::common::type::DEVICE_CPU :
	case cci::common::type::DEVICE_MCORE :


		uint64_t t1, t0;// timing variables
		t0 = cci::common::event::timestampInUS();
		inputImg = imread(std::string(input), -1);

		normalized = nscale::Normalization::normalization(inputImg, meanT, stdT);

		imwrite("normalized.tiff", normalized);
		t1 = cci::common::event::timestampInUS();
		std::cout << "time normalization: "<< t1 -t0 << std::endl;
//	 default	nscale::HistologicalEntities::segmentNuclei(normalized, maskMat, 220, 220, 220, 5.0, 4.0, 80, 11, 1000, 45, 30, 21, 1000, 4, 8, 8, logger, iwrite);
// ga img3		
		nscale::HistologicalEntities::segmentNuclei(normalized, maskMat, 220, 220, 220, 4.0, 3.5, 44, 5, 9000, 31, 20, 5, 1150, 4, 4, 4, logger, iwrite);

		std::cout << " mask : "<< std::string(mask) << std::endl; 
		imwrite(std::string(mask), maskMat);
		break;
	case cci::common::type::DEVICE_GPU :
		nscale::gpu::HistologicalEntities::segmentNuclei(std::string(input), std::string(mask), compcount, bbox, NULL, logger, iwrite);
		break;
	default :
		break;
	}
	if (bbox != NULL) free(bbox);
	else printf("WHY IS BBOX NULL?\n");

	if (logger) logger->endSession();
	if (iwrite)	delete iwrite;

}

int main (int argc, char **argv){
    	printf("single threaded because logger is not multi-threaded\n");

    	// parse the input
    	int modecode;
    	std::string imageName, outDir, hostname;
    	::cciutils::SimpleCSVLogger *logger = NULL;
    	bool debug = false;
    	int status = parseInput(argc, argv, modecode, imageName, outDir, debug, logger);

    	printf("logger created ? %s \n", logger == NULL ? "no" : "yes");
    	if (status != 0) return status;

    	uint64_t t0 = 0, t1 = 0, t2 = 0;
    	t1 = cci::common::event::timestampInUS();

    	// first get the list of files to process
       	std::vector<std::string> filenames;
    	std::vector<std::string> seg_output;
    	std::vector<std::string> bounds_output;

    	t0 = cci::common::event::timestampInUS();
    	getFiles(imageName, outDir, filenames, seg_output, bounds_output);

    	t1 = cci::common::event::timestampInUS();
    	printf("file read took %lu us\n", t1 - t0);

    	int total = filenames.size();
    	int i = 0;
    	while (i < total) {
    		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, debug, logger);
    		++i;
    	}
		t2 = cci::common::event::timestampInUS();
		printf("FINISHED in %lu us\n", t2 - t1);

		if (logger) delete logger;

    	return 0;
}
