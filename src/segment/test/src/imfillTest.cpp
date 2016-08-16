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


using namespace cv;


int main (int argc, char **argv){

	// imfill test:
	// example from matlab doc for imfill.
	/*
	 *     Fill in the background of a binary image from a specified starting
    location:

        BW1 = logical([1 0 0 0 0 0 0 0
                       1 1 1 1 1 0 0 0
                       1 0 0 0 1 0 1 0
                       1 0 0 0 1 1 1 0
                       1 1 1 1 0 1 1 1
                       1 0 0 1 1 0 1 0
                       1 0 0 0 1 0 1 0
                       1 0 0 0 1 1 1 0]);
        BW2 = imfill(BW1,[3 3],8)

    Fill in the holes of a binary image:

        BW4 = im2bw(imread('coins.png'));
        BW5 = imfill(BW4,'holes');
        figure, imshow(BW4), figure, imshow(BW5)

    Fill in the holes of an intensity image:

        I = imread('tire.tif');
        I2 = imfill(I,'holes');
        figure, imshow(I), figure, imshow(I2)

	 */
	Mat imfilldata = imread("in-fillHolesDump.ppm", -1);
//	Mat imfillinput = repeat(imfilldata, 512, 512);
//	Mat seeds = Mat::zeros(imfilldata.size(), CV_8U);
//	seeds.ptr(2)[2] = 1;
//	Mat imfillseeds = repeat(seeds, 512, 512);
//
//	uint64_t t1 = cci::common::event::timestampInUS();
//	Mat filled = nscale::imfill<unsigned char>(imfillinput, imfillseeds, true, 8);
//	uint64_t t2 = cci::common::event::timestampInUS();
//	std::cout << "imfill took " << t2-t1 << "ms" << std::endl;
//	imwrite("test/out-imfilled.pbm", filled);
//
	Mat filled;
	// imfill holes
	uint64_t t1 = cci::common::event::timestampInUS();
	filled = nscale::imfillHoles<unsigned char>(imfilldata, true, 8);
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "imfill holes took " << t2-t1 << "ms" << std::endl;
	imwrite("out-holesfilledCPU.pbm", filled);

#ifdef WITH_CUDA
	GpuMat input(imfilldata);
	Stream stream;
	t1 = cci::common::event::timestampInUS();
	GpuMat g_filled = nscale::gpu::imfillHoles<unsigned char>(input, true, 8, stream);
	t2 = cci::common::event::timestampInUS();


	std::cout << "imfill holes gpu took " << t2-t1 << "ms" << std::endl;

	Mat filledGPU(g_filled.size(), g_filled.type());

	stream.enqueueDownload(g_filled, filledGPU);
	stream.waitForCompletion();

	imwrite("out-holesfilledGPU.pbm", filledGPU);
#endif

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

