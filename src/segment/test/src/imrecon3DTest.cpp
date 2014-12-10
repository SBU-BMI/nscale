/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"
#include <stdio.h>

#include "opencv2/gpu/gpu.hpp"



using namespace cv;
using namespace cv::gpu;
using namespace std;

int main (int argc, char **argv){
	// test perfromance of imreconstruct.

	if(argc != 3){
		std::cout << "Usage: ./imrecon3DTest <input-path> <numberOfSlices>" << std::endl;
		exit(1);
	}
	std::string inputPathName = argv[1];
	int nSlices = atoi(argv[2]);
	assert(nSlices > 0);

	std::vector<Mat> mask;
	std::vector<Mat> marker;
	// assuming that input-path contains prefix of file
	// name. We add slice number and ".tif" extension to that name.
	for(int i = 1; i <= nSlices; i++){
		std::string sliceFilename = inputPathName;
		stringstream ss;
		ss << i;
		sliceFilename.append(ss.str());
		sliceFilename.append(".tif");
		Mat aux = imread(sliceFilename, -1);
	/*	std::cout << "FileName: "<< sliceFilename << std::endl;
		unsigned short int max = 0;
		for (int y = 0; y < aux.rows; ++y) {
			unsigned short int *ptr = aux.ptr<unsigned short int>(y);
			for (int x = 0; x < aux.cols; ++x) {
				if(ptr[x] > max) max =ptr[x];
			}
		}
		std::cout << "max: "<< max << std::endl;*/
		mask.push_back(aux);
		marker.push_back(aux.clone()-11);
	}

	
/*	for(int i = 0; i < mask.size(); i++){
		std::cout << "Index: "<< i << " rows: "<< mask[i].rows << " cols: "<< mask[i].cols << " channels: "<< mask[i].channels() << " is usint? " << (mask[i].type() == CV_16U)<< std::endl;

	}*/
	// vector<Mat>;
	std::vector<Mat> recon, imhmax;
	uint64_t t1, t2;

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstruct3D<unsigned short int>(marker, mask, 6);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu recon3D-6point took " << (t2-t1)/1000 << "ms" << std::endl;

	// dump output
	for(int i = 0; i < recon.size(); i++){
		std::string outFileName = "output";
		stringstream ss;
		ss << (i+1);
		outFileName.append(ss.str());
		outFileName.append(".tif");
		imwrite(outFileName, recon[i]);
	}

	t1 = cci::common::event::timestampInUS();
	imhmax = nscale::imhmax3D<unsigned short int>(mask,12,  6);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu imhmax3D-6point took " << (t2-t1)/1000 << "ms" << std::endl;

	// dump output
	for(int i = 0; i < recon.size(); i++){
		std::string outFileName = "outputimhmax";
		stringstream ss;
		ss << (i+1);
		outFileName.append(ss.str());
		outFileName.append(".tif");
		imwrite(outFileName, imhmax[i]);
	}

/*	imwrite("test/out-recon4-cpu.ppm", recon);

	Mat markerUSINT, maskUSINT;
	marker.convertTo(markerUSINT, CV_16UC1, 1, 0);
	mask.convertTo(maskUSINT, CV_16UC1, 1, 0);

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstruct<unsigned short int>(markerUSINT, maskUSINT, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu recon4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon4-cpu-usint.ppm", recon);
	recon.convertTo(recon, CV_8UC1, 1, 0);
	imwrite("test/out-recon4-cpu-usint-char.ppm", recon);*/




	return 0;
}
