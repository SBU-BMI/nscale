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
	Mat imfilldata = imread("test/imfillTest.pbm", 0);
	Mat imfillinput = repeat(imfilldata, 512, 512);
	Mat seeds = Mat::zeros(imfilldata.size(), CV_8U);
	seeds.ptr(2)[2] = 1;
	Mat imfillseeds = repeat(seeds, 512, 512);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat filled = nscale::imfill<uchar>(imfillinput, imfillseeds, true, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "imfill took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-imfilled.pbm", filled);

	// imfill holes
	t1 = cciutils::ClockGetTime();
	filled = nscale::imfillHoles<uchar>(imfillinput, true, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill holes took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-holesfilled.pbm", filled);


	t1 = cciutils::ClockGetTime();
	filled = nscale::imfill<uchar>(imfillinput, imfillseeds, true, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill 4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-imfilled4.pbm", filled);

	// imfill holes
	t1 = cciutils::ClockGetTime();
	filled = nscale::imfillHoles<uchar>(imfillinput, true, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill holes4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-holesfilled4.pbm", filled);

	// grayscale fill holes
	imfilldata = imread("test/tire.tif", 0);
	imfillinput = repeat(imfilldata, 20, 17);

	t1 = cciutils::ClockGetTime();
	filled = nscale::imfillHoles<uchar>(imfillinput, false, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill holes gray took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-holesfilled-gray.ppm", filled);

	t1 = cciutils::ClockGetTime();
	filled = nscale::imfillHoles<uchar>(imfillinput, false, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "imfill holes gray4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-holesfilled-gray4.ppm", filled);


	// bwselect testing
	/*
	 *         BW1 = imread('text.png');
        c = [126 187 11];
        r = [34 172 20];
        BW2 = bwselect(BW1,c,r,4);
	 *
	 */
	imfilldata = imread("test/text.png", 0);
	imfillinput = repeat(imfilldata, 16, 16);
	seeds = Mat::zeros(imfilldata.size(), CV_8U);
	seeds.ptr(125)[33] = 1;
	seeds.ptr(186)[171] = 1;
	seeds.ptr(10)[19] = 1;
	imfillseeds = repeat(seeds, 16, 16);

	t1 = cciutils::ClockGetTime();
	Mat bwselected = nscale::bwselect<uchar>(imfillinput, imfillseeds, 8);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwselect took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwselected.pbm", bwselected);

	t1 = cciutils::ClockGetTime();
	bwselected = nscale::bwselect<uchar>(imfillinput, imfillseeds, 4);
	t2 = cciutils::ClockGetTime();
	std::cout << "bwselect4 took " << t2-t1 << "ms" << std::endl;
	imwrite("test/out-bwselected4.pbm", bwselected);



//	waitKey();

	return 0;
}

