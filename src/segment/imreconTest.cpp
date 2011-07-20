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
#include "MorphologicOperations.h"
#include "ImageOperations.h"
#include "utils.h"

using namespace cv;


int main (int argc, char **argv){
	// test perfromance of imreconstruct.
	Mat_<uchar> mask(4096,4096);
	randn(mask, Scalar::all(128), Scalar::all(30));
	imwrite("/home/tcpan/PhD/path/mask.ppm", mask);
	Mat el = getStructuringElement(MORPH_RECT, Size(7,7));
	Mat_<uchar> marker(4096,4096);
	morphologyEx(mask, marker, CV_MOP_OPEN, el);
	imwrite("/home/tcpan/PhD/path/marker.ppm", marker);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat_<uchar> recon = nscale::imreconstruct(marker, mask, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "recon took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/recon.ppm", recon);

	t1 = cciutils::ClockGetTime();
	Mat_<uchar> recon3 = nscale::imreconstructScan(marker, mask, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon Scan took " << t2-t1 << "ms" << std::endl;
	imwrite("/home/tcpan/PhD/path/reconScan.ppm", recon3);


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
	Mat filled = nscale::imfill<uchar>(255 - maskb, markerb, 8);
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
