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
#include <string>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "UtilsLogger.h"
#include "UtilsCVImageIO.h"


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
		modecode = cciutils::DEVICE_CPU;
		// get core count


	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count


	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
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
	std::string suffix;
	suffix.assign(".tif");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(imageName, filenames);
	std::string dirname;
	if (filenames.size() == 1) {
		dirname = imageName.substr(0, imageName.find_last_of("/\\"));
	} else {
		dirname = imageName;
	}


	std::string temp, tempdir;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
			// generate the output file name
		temp = futils.replaceExt(filenames[i], ".tif", ".mask.pbm");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
		seg_output.push_back(temp);
		// generate the bounds output file name
		temp = futils.replaceExt(filenames[i], ".tif", ".bounds.csv");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
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


    	FileUtils fu(".mask.pbm");
    	std::string fmask(mask);

        std::string prefix = fu.replaceExt(fmask, ".mask.pbm", "");
        std::string suffix;
        suffix.assign(".mask.pbm");
    	iwrite = new ::cciutils::cv::IntermediateResultWriter(prefix, suffix, stages);

    	//printf("creating a debugger\n");
    }


    if (logger) {
		logger->log("filename", std::string(input));
		logger->log("type", modecode);
    }



	switch (modecode) {
	case cciutils::DEVICE_CPU :
	case cciutils::DEVICE_MCORE :
		nscale::HistologicalEntities::segmentNuclei(std::string(input), std::string(mask), logger, iwrite);
		break;
	case cciutils::DEVICE_GPU :
		nscale::gpu::HistologicalEntities::segmentNuclei(std::string(input), std::string(mask), NULL, logger, iwrite);
		break;
	default :
		break;
	}

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
    	t1 = cciutils::ClockGetTime();

    	// first get the list of files to process
       	std::vector<std::string> filenames;
    	std::vector<std::string> seg_output;
    	std::vector<std::string> bounds_output;

    	t0 = cciutils::ClockGetTime();
    	getFiles(imageName, outDir, filenames, seg_output, bounds_output);

    	t1 = cciutils::ClockGetTime();
    	printf("file read took %lu us\n", t1 - t0);

    	int total = filenames.size();
    	int i = 0;


    	while (i < total) {
    		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, debug, logger);
    		++i;
    	}
		t2 = cciutils::ClockGetTime();
		printf("FINISHED in %lu us\n", t2 - t1);

		if (logger) delete logger;

    	return 0;
}
