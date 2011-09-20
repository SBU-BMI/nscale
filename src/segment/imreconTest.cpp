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

	Mat maskb = mask > (0.8 * 255) ;
	imwrite("test/in-maskb.pbm", maskb);

	Mat markerb = mask > (0.9 * 255);
	imwrite("test/in-markerb.pbm", markerb);

	Mat mask2 = imread("DownhillFilter/Loop.pgm", 0);
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", mask2);

	Mat marker2(Size(256,256), CV_8U);
	marker2.ptr<uchar>(112)[93] = 255;


	Mat recon, recon2;
	uint64_t t1, t2;

	Stream stream;
	GpuMat g_marker;
	GpuMat g_mask, g_recon;



	stream.enqueueUpload(marker2, g_marker);
	stream.enqueueUpload(mask2, g_mask);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 8, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu recon loop took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon2);
	imwrite("test/out-reconLoop-gpu.pbm", recon2);
	g_recon.release();

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu recon loop 4 took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon2);
	imwrite("test/out-reconLoop4-gpu.pbm", recon2);
	g_recon.release();

//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 8, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "gpu recon2 loop took " << t2-t1 << "ms" << std::endl;
//	g_recon.download(recon2);
//	imwrite("test/out-recon2Loop-gpu.pbm", recon2);
//	g_recon.release();
//
//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 4, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "gpu recon2 loop 4 took " << t2-t1 << "ms" << std::endl;
//	g_recon.download(recon2);
//	imwrite("test/out-recon2Loop4-gpu.pbm", recon2);
//	g_recon.release();


	g_marker.release();
	g_mask.release();


	stream.enqueueUpload(marker, g_marker);
	stream.enqueueUpload(mask, g_mask);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;

//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 8, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "gpu recon2 took " << t2-t1 << "ms" << std::endl;
//	g_recon.download(recon);
//	imwrite("test/out-recon2-gpu.ppm", recon);
//	g_recon.release();
//
//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 4, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "gpu recon24 took " << t2-t1 << "ms" << std::endl;
//	g_recon.download(recon);
//	imwrite("test/out-recon24-gpu.ppm", recon);
//	g_recon.release();

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 8, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu recon took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon);
	imwrite("test/out-recon-gpu.ppm", recon);
	g_recon.release();

	t1 = cciutils::ClockGetTime();
	g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu recon4 took " << t2-t1 << "ms" << std::endl;
	g_recon.download(recon);
	imwrite("test/out-recon4-gpu.ppm", recon);
	g_recon.release();

	g_marker.release();
	g_mask.release();


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



	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstruct<uchar>(marker, mask, 8);
	t2 = cciutils::ClockGetTime();
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




	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstruct<uchar>(marker2, mask2, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon Loop took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconuL.ppm", recon);

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstruct<uchar>(marker2, mask2, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "recon Loop 4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-reconuL4.ppm", recon);

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






	//waitKey();



	return 0;
}

