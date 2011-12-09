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
#include <vector>
#include <string>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>


using namespace cv;


int main (int argc, char **argv){
    printf( " MPI disabled\n");

    // relevant to head node only
    std::vector<std::string> filenames;
	std::vector<std::string> seg_output;
	std::vector<std::string> bounds_output;
	int dataCount;

	// relevant to all nodes
	int modecode = 0;
	uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	std::string fin, fmask, fbound;

	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	std::string imagename(argv[1]);
	std::string outDir(argv[2]);
	const char* mode = argc > 4 ? argv[4] : "cpu";

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
		if (argc > 5) {
			gpu::setDevice(atoi(argv[5]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	cciutils::SimpleCSVLogger logger(argv[3]);
	logger.consoleOn();
	logger.on();

		// check to see if it's a directory or a file
		std::string suffix;
		suffix.assign(".tif");

		FileUtils futils(suffix);
		futils.traverseDirectoryRecursive(imagename, filenames);
		std::string dirname;
		if (filenames.size() == 1) {
			dirname = filenames[0].substr(0, filenames[0].find_last_of("/\\"));
		} else {
			dirname = imagename;
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
		dataCount= filenames.size();


		t3 = cciutils::ClockGetTime();


    for (unsigned int i = 0; i < dataCount; ++i) {
		fin = filenames[i];
		fmask = seg_output[i];
		fbound = bounds_output[i];
//		printf("in seq seg loop with rank %d, loop %d\n", rank, i);

		int tid = 0;

//  	std::cout << outfile << std::endl;

		t1 = cciutils::ClockGetTime();

		int status;

		logger.log("run-id", argv[3]);
		logger.log("filename", fin);
		logger.log("type", mode);
		

			switch (modecode) {
			case cciutils::DEVICE_CPU :
			case cciutils::DEVICE_MCORE :
				status = nscale::HistologicalEntities::segmentNuclei(fin, fmask, &logger);
				break;
			case cciutils::DEVICE_GPU :
				status = nscale::gpu::HistologicalEntities::segmentNuclei(fin, fmask, &logger);
				break;
			default :
				break;
			}
		logger.endSession();

		t2 = cciutils::ClockGetTime();
		printf(" segment %lu us, in %s\n", t2-t1, fin.c_str());
//		std::cout << rank <<"::" << tid << ":" << t2-t1 << " us, in " << fin << ", out " << fmask << std::endl;
}

    	t4 = cciutils::ClockGetTime();
		printf("**** Segment took %lu us\n", t4-t3);
	//	std::cout << "**** Segment took " << t4-t3 << " us" << std::endl;




	return 0;
}


