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
#include <time.h>
#include "MorphologicOperations.h"
#include "utils.h"

namespace {

using ::cv;


int main (int argc, char **argv){
	// test perfromance of imreconstruct.
	Mat mask(Size(4096,4096), CV_8U);
	randn(mask, Scalar::all(128), Scalar::all(30));
	imwrite("/home/tcpan/PhD/path/mask.ppm", mask);
	Mat el = getStructuringElement(MORPH_RECT, Size(7,7));
	Mat marker(Size(4096,4096), CV_8U);
	morphologyEx(mask, marker, CV_MOP_OPEN, el);
	imwrite("/home/tcpan/PhD/path/marker.ppm", marker);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat recon = nscale::imreconstruct<uchar>(marker, mask, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "recon took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/recon.ppm", recon);


	Mat maskb = mask > (0.8 * 255) ;
	imwrite("/home/tcpan/PhD/path/maskb.pbm", maskb);

	Mat markerb = mask > (0.9 * 255);
	imwrite("/home/tcpan/PhD/path/markerb.pbm", markerb);

	t1 = cciutils::ClockGetTime();
	Mat recon2 = nscale::imreconstructBinary<uchar>(markerb, maskb, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconBinary took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/reconBin.pbm", recon2);


	// imfill testing
	t1 = cciutils::ClockGetTime();
	Mat filled = nscale::imfill<uchar>(255 - maskb, markerb, true, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/imfilled.pbm", filled);


	// bwselect testing
	t1 = cciutils::ClockGetTime();
	Mat bwselected = nscale::bwselect<uchar>(maskb, markerb, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwselect took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/bwselected.pbm", bwselected);


	waitKey();

	return 0;
}

}
