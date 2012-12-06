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


using namespace cv;


int main (int argc, char **argv){

	Mat imfilldata = imread("in-imhmin.ppm", -1);
	if(imfilldata.data == NULL){
		std::cout << "Error reading input data"<< std::endl;
		exit(1);
	}

	std::cout << "in-imhimn. size(type) = "<< (imfilldata.type() == CV_32FC1) << std::endl;

	Mat imhminout;
	// imfill holes
	uint64_t t1 = cci::common::event::timestampInUS();
	imhminout = nscale::imhmin<float>(imfilldata, 1.0f, 8 );
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "imhmin holes took " << t2-t1 << "ms" << std::endl;
	imwrite("out-imhmin CPU.pbm", imhminout);

	GpuMat input(imfilldata);
	Stream stream;
	t1 = cci::common::event::timestampInUS();
	GpuMat g_outmin = nscale::gpu::imhmin<float>(input, 1.0f, 8, stream);
	t2 = cci::common::event::timestampInUS();


	std::cout << "imhmin holes gpu took " << t2-t1 << "ms" << std::endl;

	Mat imhminout_gpu(g_outmin.size(), g_outmin.type());

	stream.enqueueDownload(g_outmin, imhminout_gpu);
	stream.waitForCompletion();

	imwrite("out-imhminGPU.pbm", imhminout_gpu);

//	t1 = cci::common::event::timestampInUS();
//	filled = nscale::imfill<unsigned char>(imfillinput, imfillseeds, true, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "imfill 4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-imfilled4.pbm", filled);
//
//	// imfill holes
//	t1 = cci::common::event::timestampInUS();
//	filled = nscale::imfillHoles<unsigned char>(imfillinput, true, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "imfill holes4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-holesfilled4.pbm", filled);
//
//	// grayscale fill holes
//	imfilldata = imread("test/tire.tif", 0);
//	imfillinput = repeat(imfilldata, 20, 17);
//
//	t1 = cci::common::event::timestampInUS();
//	filled = nscale::imfillHoles<unsigned char>(imfillinput, false, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "imfill holes gray took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-holesfilled-gray.ppm", filled);
//
//	t1 = cci::common::event::timestampInUS();
//	filled = nscale::imfillHoles<unsigned char>(imfillinput, false, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "imfill holes gray4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-holesfilled-gray4.ppm", filled);
//
//
//	// bwselect testing
//	/*
//	 *         BW1 = imread('text.png');
//        c = [126 187 11];
//        r = [34 172 20];
//        BW2 = bwselect(BW1,c,r,4);
//	 *
//	 */
//	imfilldata = imread("/home/tcpan/PhD/path/src/nscale/src/segment/test/text.png", 0);
//	imfillinput = repeat(imfilldata, 16, 16);
//	seeds = Mat::zeros(imfilldata.size(), CV_8U);
//	seeds.ptr(125)[33] = 1;
//	seeds.ptr(186)[171] = 1;
//	seeds.ptr(10)[19] = 1;
//	imfillseeds = repeat(seeds, 16, 16);
//	t1 = cci::common::event::timestampInUS();
//	Mat bwselected = nscale::bwselect<unsigned char>(imfillinput, imfillseeds, 8);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "bwselect took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-bwselected.pbm", bwselected);
//
//	t1 = cci::common::event::timestampInUS();
//	Mat bwselected2 = nscale::bwselect<unsigned char>(imfillinput, imfillseeds, 4);
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "bwselect4 took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-bwselected4.pbm", bwselected2);
//
//#if defined (WITH_CUDA)
//	GpuMat g_imfillinput(imfillinput.size(), imfillinput.type());
//	Stream stream;
//	stream.enqueueUpload(imfillinput, g_imfillinput);
//	GpuMat g_imfillseeds(imfillseeds.size(), imfillseeds.type());
//	stream.enqueueUpload(imfillseeds, g_imfillseeds);
//	stream.waitForCompletion();
//	t1 = cci::common::event::timestampInUS();
//	GpuMat g_bwselected = nscale::gpu::bwselect<unsigned char>(g_imfillinput, g_imfillseeds, 8, stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "bwselect gpu took " << t2-t1 << "ms" << std::endl;
//	Mat bwselected3(g_bwselected.size(), g_bwselected.type());
//	stream.enqueueDownload(g_bwselected, bwselected3);
//	stream.waitForCompletion();
//	imwrite("test/out-bwselected-gpu.pbm", bwselected3);
//
//	t1 = cci::common::event::timestampInUS();
//	GpuMat g_bwselected2 = nscale::gpu::bwselect<unsigned char>(g_imfillinput, g_imfillseeds, 4, stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "bwselect4 gpu took " << t2-t1 << "ms" << std::endl;
//	Mat bwselected4(g_bwselected2.size(), g_bwselected2.type());
//	stream.enqueueDownload(g_bwselected2, bwselected4);
//	stream.waitForCompletion();
//	imwrite("test/out-bwselected4-gpu.pbm", bwselected4);
//#endif



//	waitKey();

	return 0;
}

