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

#include <omp.h>

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
	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu [numCores] | mcore [numCores] | gpu [id]]" << std::endl;
		return -1;
	}
	string imagename(argv[1]);
	string outDir(argv[2]);
	const char* runid = argv[3];
	const char* mode = argc > 4 ? argv[4] : "cpu";

	int modecode = 0;
	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
		}
	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
		}
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			std::cout << "gpu requested, but no gpu available.  please use cpu or mcore option."  << std::endl;
			return -2;
		}
		if (argc > 5) {
			gpu::setDevice(atoi(argv[5]));
		}
		std::cout << " number of cuda enabled devices = " << gpu::getCudaEnabledDeviceCount() << std::endl;
	} else {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}

	// check to see if it's a directory or a file
    DIR *dir;
    std::vector<std::string> filenames;
    std::string suffix;
    suffix.assign(".tif");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(imagename, filenames);
	std::string dirname;
	if (filenames.size() == 1) {
		dirname = imagename.substr(0, imagename.find_last_of("/\\"));
	} else {
		dirname = imagename;
	}
//	std::cout << "dirname " << dirname << "."<< std::endl;

	uint64_t t3 = cciutils::ClockGetTime();
#pragma omp parallel for
    for (int i = 0; i < filenames.size(); ++i) {

    	int tid = omp_get_thread_num();

    	// generate the output file name
    	string outfile = futils.replaceExt(filenames[i], ".tif", ".mask.pbm");
    	outfile = futils.replaceDir(outfile, dirname, outDir);
//  	std::cout << outfile << std::endl;

		uint64_t t1 = cciutils::ClockGetTime();

		int status;

		switch (modecode) {
		case cciutils::DEVICE_CPU :
		case cciutils::DEVICE_MCORE :
			status = nscale::HistologicalEntities::segmentNuclei(filenames[i], outfile);
			break;
		case cciutils::DEVICE_GPU :
			status = nscale::gpu::HistologicalEntities::segmentNuclei(filenames[i], outfile);
			break;
		default :
			break;
		}
		uint64_t t2 = cciutils::ClockGetTime();
		std::cout << t2-t1 << " us, in " << filenames[i] << ", out " << outfile << std::endl;


#ifdef PRINT_CONTOUR_TEXT
		Mat output = imread(outfile);
	// for Lee and Jun to test the contour correctness.
	Mat temp = Mat::zeros(output.size() + Size(2,2), output.type());
	copyMakeBorder(output, temp, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	//TODO: TEMP std::cout << "num contours = " << contours.size() << std::endl;

	// generate the output file name
	string boundary = futils.replaceExt(filenames[i], ".tif", ".boundary.txt");
	boundary = futils.replaceDir(boundary, dirname, outDir);


	std::ofstream fid(boundary.c_str());
	int counter = 0;
	if (contours.size() > 0) {
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
			fid << idx << ": ";
			for (int ptc = 0; ptc < contours[idx].size(); ++ptc) {
				fid << contours[idx][ptc].x << "," << contours[idx][ptc].y << "; ";
			}
			fid << std::endl;
		}
		++counter;
	}
	fid.close();

#endif



    }

    uint64_t t4 = cciutils::ClockGetTime();
	std::cout << "**** SEGMENTATION took " << t4-t3 << " us" << std::endl;

//	waitKey();

	return 0;
}


