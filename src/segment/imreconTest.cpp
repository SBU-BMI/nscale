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


using namespace cv;


int main (int argc, char **argv){
	// test perfromance of imreconstruct.
	Mat mask(Size(4096,4096), CV_8U);
	randn(mask, Scalar::all(128), Scalar::all(30));
	imwrite("test/in-mask.ppm", mask);
	Mat el = getStructuringElement(MORPH_RECT, Size(7,7));
	Mat marker(Size(4096,4096), CV_8U);
	morphologyEx(mask, marker, CV_MOP_OPEN, el);
	imwrite("test/in-marker.ppm", marker);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat recon = nscale::imreconstruct<uchar>(marker, mask, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "recon took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon.ppm", recon);


	Mat maskb = mask > (0.8 * 255) ;
	imwrite("test/in-maskb.pbm", maskb);

	Mat markerb = mask > (0.9 * 255);
	imwrite("test/in-markerb.pbm", markerb);

	t1 = cciutils::ClockGetTime();
	Mat recon2 = nscale::imreconstructBinary<uchar>(markerb, maskb, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconBinary took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconBin.pbm", recon2);



	// bwlabel testing
	t1 = cciutils::ClockGetTime();
	Mat bwselected = nscale::bwlabel(maskb, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwlabel took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwlabel.pbm", bwselected);


	// bwareaopen testing
	t1 = cciutils::ClockGetTime();
	Mat bwareaopen = nscale::bwareaopen<uchar>(maskb, 30, 100, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen 30-100 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-30-100.pbm", bwareaopen);
	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 0, 30, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen 0-30 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-0-30.pbm", bwareaopen);	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 100, 255, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen 100-255 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-100-255.pbm", bwareaopen);




	waitKey();

	return 0;
}

