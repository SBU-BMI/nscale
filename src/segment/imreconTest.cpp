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
#include <stdio.h>

#include "opencv2/gpu/gpu.hpp"



using namespace cv;
using namespace cv::gpu;


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

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstruct<uchar>(marker, mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon4.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker, mask, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconUchar took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconu.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker, mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconUchar 4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconu4.ppm", recon);


	Stream stream;
	GpuMat g_marker;
	stream.enqueueUpload(marker, g_marker);
	GpuMat g_mask, g_recon;
	stream.enqueueUpload(mask, g_mask);

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstructInt<uchar>(g_marker, g_mask, 8, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "recon took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon);
	imwrite("test/out-recon-gpu.ppm", recon);
	g_recon.release();

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstructInt<uchar>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "recon4 took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon);
	imwrite("test/out-recon4-gpu.ppm", recon);
	g_recon.release();
	g_marker.release();
	g_mask.release();

	Mat mask2 = imread("DownhillFilter/Loop.pgm", 0);
	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
	imshow("orig image", mask2);

	Mat marker2(Size(256,256), CV_8U);
	marker2.ptr<uchar>(112)[93] = 255;

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker2, mask2, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconUcharLoop took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconuL.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker2, mask2, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconUcharLoop 4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconuL4.ppm", recon);



	Mat maskb = mask > (0.8 * 255) ;
	imwrite("test/in-maskb.pbm", maskb);

	Mat markerb = mask > (0.9 * 255);
	imwrite("test/in-markerb.pbm", markerb);

	Mat recon2;
	t1 = cciutils::ClockGetTime();
	recon2 = nscale::imreconstructBinary<uchar>(markerb, maskb, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon Binary took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconBin-gpu.pbm", recon2);
	g_recon.release();

	t1 = cciutils::ClockGetTime();
	recon2 = nscale::imreconstructBinary<uchar>(markerb, maskb, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "reconBinary4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconBin4-gpu.pbm", recon2);


	stream.enqueueUpload(markerb, g_marker);
	stream.enqueueUpload(maskb, g_mask);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;


	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstructBinary<uchar>(g_marker, g_mask, 8, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu recon Binary took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon2);
	imwrite("test/out-reconBin-gpu.pbm", recon2);
	g_recon.release();

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstructBinary<uchar>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu reconBinary4 took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon2);
	imwrite("test/out-reconBin4-gpu.pbm", recon2);
	g_recon.release();
	g_marker.release();
	g_mask.release();





	//Mat imfilldata = imread("test/text.png", 0) > 0;
	//maskb = repeat(imfilldata, 16, 16);
	maskb = imread("test/sizePhantom.ppm", 0) > 0;

	// bwareaopen testing
	t1 = cciutils::ClockGetTime();
	Mat bwareaopen = nscale::bwareaopen<uchar>(maskb, 100, 500, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen mid took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-mid.pbm", bwareaopen);
	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 1, 100, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen small took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-small.pbm", bwareaopen);
	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 500, std::numeric_limits<int>::max(), 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen large took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen-large.pbm", bwareaopen);


	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 100, 500, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen4 mid took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen4-mid.pbm", bwareaopen);
	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 1, 100, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen4 small took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen4-small.pbm", bwareaopen);
	t1 = cciutils::ClockGetTime();
	bwareaopen = nscale::bwareaopen<uchar>(maskb, 500, std::numeric_limits<int>::max(), 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwareaopen4 large took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwareaopen4-large.pbm", bwareaopen);

/*
	// bwlabel testing
	t1 = cciutils::ClockGetTime();
	Mat bwselected = nscale::bwlabel(maskb, false, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwlabel took " << t2-t1 << "ms" << std::endl;
	// write the raw image
	cciutils::cv::imwriteRaw("test/out-bwlabel", bwselected);
*/



	waitKey();

	return 0;
}

