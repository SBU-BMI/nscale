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
#include "utils.h"

using namespace cv;


int main (int argc, char **argv){
	// test perfromance of imreconstruct.
	Mat_<uchar> mask = Mat_<uchar>(4096,4096);
	randn(mask, Scalar::all(128), Scalar::all(30));
	imwrite("/home/tcpan/PhD/path/mask.pbm", mask);
	Mat el = getStructuringElement(MORPH_RECT, Size(7,7));
	Mat_<uchar> marker = Mat_<uchar>(4096,4096);
	morphologyEx(mask, marker, CV_MOP_OPEN, el);
	imwrite("/home/tcpan/PhD/path/marker.pbm", marker);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat_<uchar> recon = nscale::imreconstruct(mask, marker, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "recon took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/recon.pbm", recon);


	t1 = cciutils::ClockGetTime();
	Mat_<uchar> recon2 = nscale::imreconstruct2(mask, marker, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon2 took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/recon2.pbm", recon2);

//	waitKey();

	return 0;
}
