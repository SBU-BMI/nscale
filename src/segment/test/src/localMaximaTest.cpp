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
#include "utils.h"


using namespace cv;


int main (int argc, char **argv){
	// test perfromance of imreconstruct.
	Mat mask(Size(4096,4096), CV_8U);
	randn(mask, Scalar::all(128), Scalar::all(30));
	imwrite("test/in-mask2-for-localmax.ppm", mask);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat recon = nscale::localMaxima<unsigned char>(mask, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "localmax took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-localmax.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::localMinima<unsigned char>(mask, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "localmin took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-localmin.ppm", recon);


	// imhmin test
	t1 = cciutils::ClockGetTime();
	Mat hmin = nscale::imhmin<unsigned char>(mask, 1, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin -1 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin-1.ppm", hmin);

	t1 = cciutils::ClockGetTime();
	hmin = nscale::imhmin<unsigned char>(mask, 2, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin -2 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin-2.ppm", hmin);

	t1 = cciutils::ClockGetTime();
	hmin = nscale::imhmin<unsigned char>(mask, 3, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin -3 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin-3.ppm", hmin);



	t1 = cciutils::ClockGetTime();
	recon = nscale::localMaxima<unsigned char>(mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "localmax4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-localmax4.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::localMinima<unsigned char>(mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "localmin4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-localmin4.ppm", recon);


	// imhmin test
	t1 = cciutils::ClockGetTime();
	hmin = nscale::imhmin<unsigned char>(mask, 1, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin4 -1 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin4-1.ppm", hmin);

	t1 = cciutils::ClockGetTime();
	hmin = nscale::imhmin<unsigned char>(mask, 2, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin4 -2 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin4-2.ppm", hmin);

	t1 = cciutils::ClockGetTime();
	hmin = nscale::imhmin<unsigned char>(mask, 3, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "hmin4 -3 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-hmin4-3.ppm", hmin);


	return 0;
}

