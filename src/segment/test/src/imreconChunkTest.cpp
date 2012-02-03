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
#include <string>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "utils.h"
#include <stdio.h>

#include "opencv2/gpu/gpu.hpp"



using namespace cv;
using namespace cv::gpu;

void runTest(const char* markerName, const char* maskName, bool binary, int w=-1, int b=0) {
	Mat marker, mask, recon;
	if (binary) marker = imread(markerName, 0) > 0;
	else marker = imread(markerName, 0);
	if (binary) mask = imread(maskName, 0) > 0;
	else mask = imread(maskName, 0);

//	std::cout << "testing with " << markerName << " + " << maskName << " " << (binary ? "binary" : "grayscale") << std::endl;
	std::cout << "hw,algo,binary,conn,chunk,border,time(us)" << std::endl;
	
	Stream stream;
	GpuMat g_marker, g_mask, g_recon;

	uint64_t t1, t2;
	Size s = marker.size();
;
	
/*	t1 = cciutils::ClockGetTime();
	if (binary) recon = nscale::imreconstructBinary<uchar>(marker, mask, 4);
	else recon = nscale::imreconstruct<uchar>(marker, mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "\tcpu recon 4-con took " << t2-t1 << "ms" << std::endl;
*/
	if (w == -1) {
		t1 = cciutils::ClockGetTime();
		if (binary) recon = nscale::imreconstructBinary<uchar>(marker, mask, 8);
		else recon = nscale::imreconstruct<uchar>(marker, mask, 8);
		t2 = cciutils::ClockGetTime();
		std::cout << "cpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<<s.width << ",0," << t2-t1 << std::endl;
	} else {
		t1 = cciutils::ClockGetTime();
		for (int j = 0; j < s.height; j+=w) {
			for (int i = 0; i < s.height; i+=w) {
				uint64_t t3 = cciutils::ClockGetTime();
		
				Range rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
				Range ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
				
				if (binary) recon = nscale::imreconstructBinary<uchar>(marker(rx, ry), mask(rx, ry), 8);
				else recon = nscale::imreconstruct<uchar>(marker(rx, ry), mask(rx, ry), 8);
				uint64_t t4 = cciutils::ClockGetTime();

				//std::cout << "\t\tchunk "<< i << "," << j << " took " << t4-t3 << "ms " <<  std::endl;
			}
		}
		t2 = cciutils::ClockGetTime();
		std::cout << "cpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< w << "," << b << "," << t2-t1 << std::endl;


	}
	
/*	
	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker, mask, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "\tcpu reconUChar 4-con took " << t2-t1 << "ms" << std::endl;

	t1 = cciutils::ClockGetTime();
	recon = nscale::imreconstructUChar(marker, mask, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "\tcpu reconUChar 8-con took " << t2-t1 << "ms" << std::endl;
*/
	
	
	
	
	stream.enqueueUpload(marker, g_marker);
	stream.enqueueUpload(mask, g_mask);
	stream.waitForCompletion();
//	std::cout << "\tfinished uploading to GPU" << std::endl;

//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 4, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "\tgpu recon2 4-con took " << t2-t1 << "ms" << std::endl;
//	g_recon.release();
//
//	t1 = cciutils::ClockGetTime();
//	g_recon = nscale::gpu::imreconstruct2<uchar>(g_marker, g_mask, 8, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "\tgpu recon2 8-con took " << t2-t1 << "ms" << std::endl;
//	g_recon.release();

	
/*	t1 = cciutils::ClockGetTime();
	if (binary) g_recon = nscale::gpu::imreconstructBinary<uchar>(g_marker, g_mask, 4, stream);
	else g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "\tgpu recon 4-con took " << t2-t1 << "ms" << std::endl;
	g_recon.release();
*/
	if (w == -1) {
		t1 = cciutils::ClockGetTime();
		if (binary) g_recon = nscale::gpu::imreconstructBinary<uchar>(g_marker, g_mask, 8, stream);
		else g_recon = nscale::gpu::imreconstruct<uchar>(g_marker, g_mask, 8, stream);
		stream.waitForCompletion();
		t2 = cciutils::ClockGetTime();
		std::cout << "gpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< s.width <<",0,"<< t2-t1 << std::endl;
		g_recon.release();
	} else {
		s = g_marker.size();
		t1 = cciutils::ClockGetTime();
		unsigned int iter;
		for (int j = 0; j < s.height; j+=w) {
			for (int i = 0; i < s.height; i+=w) {
				uint64_t t3 = cciutils::ClockGetTime();
		
				Range rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
				Range ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
				
				if (binary) g_recon = nscale::gpu::imreconstructBinary<uchar>(g_marker(rx, ry), g_mask(rx, ry), 8, stream, iter);
				else g_recon = nscale::gpu::imreconstruct<uchar>(g_marker(rx, ry), g_mask(rx, ry), 8, stream, iter);
				stream.waitForCompletion();
				uint64_t t4 = cciutils::ClockGetTime();

				g_recon.release();
				//std::cout << "\t\tchunk "<< i << "," << j << " took " << t4-t3 << "ms with " << iter << " iters" <<  std::endl;
			}
		}
		t2 = cciutils::ClockGetTime();
		std::cout << "gpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< w << "," << b << "," << t2-t1 << std::endl;


	}

	

	g_marker.release();
	g_mask.release();


	
}


int main (int argc, char **argv){

//	runTest("test/in-bwselect-marker.pgm", "test/in-bwselect-mask.pgm", true);
//	runTest("test/in-bwselect-marker.pgm", "test/in-bwselect-mask.pgm", false);
//	runTest("test/in-fillholes-bin-marker.pgm", "test/in-fillholes-bin-mask.pgm", true);
//	runTest("test/in-fillholes-bin-marker.pgm", "test/in-fillholes-bin-mask.pgm", false);
//	runTest("test/in-imrecon-gray-marker.pgm", "test/in-imrecon-gray-mask.pgm", true);
	runTest("test/in-imrecon-gray-marker.pgm", "test/in-imrecon-gray-mask.pgm", false);
	for (int w = 128; w < 4096; w = w* 2) {
		for (int b2 = 1; b2 <= 512 && b2 <= (w/2); b2 = b2 * 2) {
			runTest("test/in-imrecon-gray-marker.pgm", "test/in-imrecon-gray-mask.pgm", false, w, b2/2);
		}
	}
		
	return 0;
}

