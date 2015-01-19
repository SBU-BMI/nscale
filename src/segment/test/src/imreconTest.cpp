/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"
#include <stdio.h>


#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "opencv2/gpu/gpu.hpp"



using namespace cv;
using namespace cv::gpu;
using namespace std;

int main (int argc, char **argv){
	// test perfromance of imreconstruct.

	if(argc != 3){
		printf("Usage: ./imreconTest <marker> <mask>");
	}
	Mat marker = imread(argv[1], -1);
	Mat mask = imread(argv[2], -1);
	
	Mat recon;
	uint64_t t1, t2;

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu recon4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon4-cpu.ppm", recon);

	Mat markerUSINT, maskUSINT;
	marker.convertTo(markerUSINT, CV_16UC1, 1, 0);
	mask.convertTo(maskUSINT, CV_16UC1, 1, 0);

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstruct<unsigned short int>(markerUSINT, maskUSINT, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu recon4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon4-cpu-usint.ppm", recon);
	recon.convertTo(recon, CV_8UC1, 1, 0);
	imwrite("test/out-recon4-cpu-usint-char.ppm", recon);

#if defined (WITH_CUDA)
	Stream stream;
	GpuMat g_marker;
	GpuMat g_mask;

	stream.enqueueUpload(marker, g_marker);
	stream.enqueueUpload(mask, g_mask);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << " cpu recon4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-recon4-cpu.ppm", recon);

	t1 = cci::common::event::timestampInUS();
	GpuMat g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_marker, g_mask, 4, 8,stream, 14, false);
	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS();

	cout << " gpu recon4 took "<< t2-t1<< "ms" <<endl;
	g_recon.download(recon);

	imwrite("test/out-recon4-gpu.ppm", recon);
	g_recon.release();
	g_marker.release();
	g_mask.release();
#endif


//	mat markerint, maskint;
//	marker.convertto(markerint, cv_32sc1, 1, 0);
//	mask.convertto(maskint, cv_32sc1, 1, 0);
//
//#if defined (WITH_CUDA)
//	GpuMat g_marker_int, g_mask_int;
//	stream.enqueueUpload(markerInt, g_marker_int);
//	stream.enqueueUpload(maskInt, g_mask_int);
//	stream.waitForCompletion();
//#endif
//
//	// INT testing
//	t1 = cci::common::event::timestampInUS();
//	Mat reconInt = nscale::imreconstruct<int>(markerInt, maskInt, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "cpu recon int took " << t2-t1 << "ms" << std::endl;
//	//imwrite("test/out-reconint.ppm", recon);
//
//#if defined (WITH_CUDA)
//	t1 = cci::common::event::timestampInUS();
//	GpuMat g_recon_int = nscale::gpu::imreconstruct<int>(g_marker_int, g_mask_int, 8, stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "gpu recon int took " << t2-t1 << "ms" << std::endl;
//	g_recon_int.download(reconInt);
//	imwrite("test/out-reconLoop4-gpu.pbm", recon2);
//	g_recon_int.release();
//
//
//
//	vector<GpuMat> g_marker_v;
//	vector<GpuMat> g_mask_v;
//
//	int num_images = atoi(argv[1]);
//	int numFirstPass = atoi(argv[2]);
//
//	for(int i = 0; i < num_images; i++){
//		g_marker_v.push_back(g_marker);
//		g_mask_v.push_back(g_mask);
//	}
////	cout << "INIT: imrecon throughput 4"<<endl;
////	t1 = cci::common::event::timestampInUS();
////	vector<GpuMat> g_recon_vector2 = nscale::gpu::imreconstructQueueThroughput<unsigned char>(g_marker_v, g_mask_v, 4, numFirstPass,stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	cout << "END: imrecon throughput 4. Time = "<< t2-t1 <<endl;
//
//	t1 = cci::common::event::timestampInUS();
//	vector<GpuMat> g_recon_vector = nscale::gpu::imreconstructQueueThroughput<unsigned char>(g_marker_v, g_mask_v, 4, numFirstPass,stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//
//	cout << "imreconstructThroughputTime = "<< t2-t1<< " images= "<< num_images<< " passes= "<< numFirstPass <<endl;
//	for(int i = 0; i < g_recon_vector.size();i++){
//		g_recon_vector[i].download(recon2);
//
//		std::stringstream sstm;
//		sstm << "test/out-recon4-gpu-list-vector-" << i << ".ppm";
//		string out_file_name = sstm.str();
//
////		cout << "OUT FILE NAME = "<< out_file_name<<endl;
////		imwrite(out_file_name, recon2);
//		g_recon_vector[i].release();
//	}
//
//
////	g_marker.release();
////	g_mask.release();
//////
//////
//////	stream.enqueueUpload(marker, g_marker);
//////	stream.enqueueUpload(mask, g_mask);
//////	stream.waitForCompletion();
//////	std::cout << "finished uploading" << std::endl;
//
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstruct2<unsigned char>(g_marker, g_mask, 8, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu recon2 took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon);
////	imwrite("test/out-recon2-gpu.ppm", recon);
////	g_recon.release();
////
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstruct2<unsigned char>(g_marker, g_mask, 4, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu recon24 took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon);
////	imwrite("test/out-recon24-gpu.ppm", recon);
////	g_recon.release();
////
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, 8, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu recon took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon);
////	imwrite("test/out-recon-gpu.ppm", recon);
////	g_recon.release();
////
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, 4, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu recon4 took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon);
////	imwrite("test/out-recon4-gpu.ppm", recon);
////
////
////	stream.enqueueUpload(markerb, g_marker);
////	stream.enqueueUpload(maskb, g_mask);
////	stream.waitForCompletion();
////	std::cout << "finished uploading" << std::endl;
//	t1 = cci::common::event::timestampInUS();
//	g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_marker, g_mask, 4, numFirstPass,stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//
//	cout << "imreconstructQueueSpeedupTime = "<< t2-t1<<endl;
//
//	g_recon.release();
//
//	g_marker.release();
//	g_mask.release();
//#endif
//
//////
//////	t1 = cci::common::event::timestampInUS();
//////	g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, 4, stream);
//////	stream.waitForCompletion();
//////	t2 = cci::common::event::timestampInUS();
//////	std::cout << "gpu recon4 took " << t2-t1 << "ms" << std::endl;
//////	g_recon.download(recon);
//////	imwrite("test/out-recon4-gpu.ppm", recon);
//////	g_recon.release();
//////
//////	g_marker.release();
//////	g_mask.release();
//////
//////
//////	stream.enqueueUpload(markerb, g_marker);
//////	stream.enqueueUpload(maskb, g_mask);
//////	stream.waitForCompletion();
//////	std::cout << "finished uploading" << std::endl;
//
//
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstructBinary<unsigned char>(g_marker, g_mask, 8, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu recon Binary took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon2);
////	imwrite("test/out-reconBin-gpu.pbm", recon2);
////	g_recon.release();
////
////	t1 = cci::common::event::timestampInUS();
////	g_recon = nscale::gpu::imreconstructBinary<unsigned char>(g_marker, g_mask, 4, stream);
////	stream.waitForCompletion();
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "gpu reconBinary4 took " << t2-t1 << "ms" << std::endl;
////	g_recon.download(recon2);
////	imwrite("test/out-reconBin4-gpu.pbm", recon2);
////	g_recon.release();
////	g_marker.release();
////	g_mask.release();
/////
/////	t1 = cci::common::event::timestampInUS();
/////	recon = nscale::imreconstructGeorge<unsigned char>(marker, mask, 4);
/////	t2 = cci::common::event::timestampInUS();
/////	std::cout << "recon4 George took " << t2-t1 << "ms" << std::endl;
/////	imwrite("test/out-recon4-george.ppm", recon);
/////
//
//	t1 = cci::common::event::timestampInUS();
//	recon = nscale::imreconstruct<unsigned char>(marker, mask, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << " cpu recon8 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-recon8.ppm", recon);
//
////	t1 = cci::common::event::timestampInUS();
////	recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "cpu recon4 took " << t2-t1 << "ms" << std::endl;
////	imwrite("test/out-recon4.ppm", recon);
////
////	t1 = cci::common::event::timestampInUS();
////	recon = nscale::imreconstructUChar(marker, mask, 8);
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "reconUchar 8 took " << t2-t1 << "ms" << std::endl;
////	imwrite("test/out-reconu.ppm", recon);
////
////	t1 = cci::common::event::timestampInUS();
////	recon = nscale::imreconstructUChar(marker, mask, 4);
////	t2 = cci::common::event::timestampInUS();
////	std::cout << "reconUchar 4 took " << t2-t1 << "ms" << std::endl;
////	imwrite("test/out-reconu4.ppm", recon);
//
//
//
///*
//	t1 = cci::common::event::timestampInUS();
//	recon = nscale::imreconstruct<unsigned char>(marker2, mask2, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "recon Loop took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconuL.ppm", recon);
//
//	t1 = cci::common::event::timestampInUS();
//	recon = nscale::imreconstruct<unsigned char>(marker2, mask2, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "recon Loop 4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconuL4.ppm", recon);
//
//	t1 = cci::common::event::timestampInUS();
//	recon = nscale::imreconstructUChar(marker2, mask2, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "reconUcharLoop took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconuL.ppm", recon);
//
//	t1 = cci::common::event::timestampInUS();
//	recon = nscale::imreconstructUChar(marker2, mask2, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "reconUcharLoop 4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconuL4.ppm", recon);
//*/
//
//
///*	t1 = cci::common::event::timestampInUS();
//	recon2 = nscale::imreconstructBinary<unsigned char>(markerb, maskb, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "recon Binary took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconBin-gpu.pbm", recon2);
//////	g_recon.release();
//
//	t1 = cci::common::event::timestampInUS();
//	recon2 = nscale::imreconstructBinary<unsigned char>(markerb, maskb, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "reconBinary4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-reconBin4-gpu.pbm", recon2);*/
//
//
//
//
//
//
//	//waitKey();
//


	return 0;
}

