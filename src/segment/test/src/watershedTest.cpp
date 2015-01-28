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
#include "PixelOperations.h"
#include "NeighborOperations.h"

#include "Logger.h"
#include <stdio.h>


#if defined (WITH_CUDA)
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/stream_accessor.hpp"
#endif

using namespace cv;
using namespace cv::gpu;


int main(int argc, char **argv){

	std::vector<std::string> segfiles;
	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/astroII.1/astroII.1.ndpi-0000008192-0000008192-15.mask.png"));
	
	std::vector<std::string> imgfiles;
	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1/astroII.1.ndpi-0000008192-0000008192.tif"));
	
	for (int i = 0; i < segfiles.size(); ++i) {

		printf("testing file : %s\n", segfiles[i].c_str());
		Mat seg_big = imread(segfiles[i].c_str(), -1);
		Mat img = imread(imgfiles[i].c_str(), -1);
		// original
		Stream stream;

		uint64_t t1, t2;

		// distance transform:  matlab code is doing this:
		// invert the image so nuclei candidates are holes
		// compute the distance (distance of nuclei pixels to background)
		// negate the distance.  so now background is still 0, but nuclei pixels have negative distances
		// set background to -inf

		// really just want the distance map.  CV computes distance to 0.
		// background is 0 in output.
		// then invert to create basins
		Mat dist(seg_big.size(), CV_32FC1);

		// opencv: compute the distance to nearest zero
		// matlab: compute the distance to the nearest non-zero
		distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
		double mmin, mmax;
		minMaxLoc(dist, &mmin, &mmax);

		// invert and shift (make sure it's still positive)
		//dist = (mmax + 1.0) - dist;
		dist = -dist;  // appears to work better this way.

		//	cciutils::cv::imwriteRaw("test/out-dist", dist);

		// then set the background to -inf and do imhmin
		//Mat distance = Mat::zeros(dist.size(), dist.type());
		// appears to work better with -inf as background
		Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
		dist.copyTo(distance, seg_big);
		//	cciutils::cv::imwriteRaw("test/out-distance", distance);


		// then do imhmin. (prevents small regions inside bigger regions)
		Mat distance2 = nscale::imhmin<float>(distance, 1.0f, 8);

		//cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);


		/*
		 *
		 seg_big(watershed(distance2)==0) = 0;
		 seg_nonoverlap = seg_big;
		 *
		 */

		Mat minima = nscale::localMinima<float>(distance2, 8);
		// watershed is sensitive to label values.  need to relabel.
		std::vector<Vec4i> dummy;
		Mat_<int> labels = nscale::bwlabel(minima, false, 8, false, dummy);

		Mat_<int> labels2 = nscale::bwlabel2(minima, 8, true);




		Mat nuclei = Mat::zeros(img.size(), img.type());
		//	Mat distance3 = distance2 + (mmax + 1.0);
		//	Mat dist4 = Mat::zeros(distance3.size(), distance3.type());
		//	distance3.copyTo(dist4, seg_big);
		//	Mat dist5(dist4.size(), CV_8U);
		//	dist4.convertTo(dist5, CV_8U, (std::numeric_limits<unsigned char>::max() / mmax));
		//	cvtColor(dist5, nuclei, CV_GRAY2BGR);
		img.copyTo(nuclei, seg_big);

		t1 = cci::common::event::timestampInUS();

		// watershed in openCV requires labels.  input foreground > 0, 0 is background
		// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
		Mat watermask = nscale::watershed(nuclei, distance2, 8);
		//	cciutils::cv::imwriteRaw("test/out-watershed", watermask);

		t2 = cci::common::event::timestampInUS();
		std::cout << "cpu watershed loop took " << t2 - t1 << "ms" << std::endl;
		double mn, mx;
		minMaxLoc(watermask, &mn, &mx);
		watermask = (watermask - mn) * (255.0 / (mx - mn));

		imwrite("test/out-cpu-watershed-oligoIII.1-1.png", watermask);

		t1 = cci::common::event::timestampInUS();

		// watershed in openCV requires labels.  input foreground > 0, 0 is background
		// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
		watermask = nscale::watershed2(nuclei, distance2, 8);
		//	cciutils::cv::imwriteRaw("test/out-watershed", watermask);

		t2 = cci::common::event::timestampInUS();
		std::cout << "cpu watershed2 loop took " << t2 - t1 << "ms" << std::endl;

		// cpu version of watershed.
		mn, mx;
		minMaxLoc(watermask, &mn, &mx);
		watermask = (watermask - mn) * (255.0 / (mx - mn));

		imwrite("test/out-cpu-watershed-oligoIII.1-2.png", watermask);
		dist.release();
		distance.release();
		watermask.release();


#if defined (WITH_CUDA)
		// gpu version of watershed
		//Stream stream;
		GpuMat g_distance2, g_watermask, g_seg_big;
		stream.enqueueUpload(distance2, g_distance2);
		stream.enqueueUpload(seg_big, g_seg_big);
		stream.waitForCompletion();
		std::cout << "finished uploading" << std::endl;

		t1 = cci::common::event::timestampInUS();
		g_watermask = nscale::gpu::watershedDW(g_seg_big, g_distance2, -1, 8, stream);
		stream.waitForCompletion();
		t2 = cci::common::event::timestampInUS();
		std::cout << "gpu watershed DW loop took " << t2-t1 << "ms" << std::endl;

		Mat temp(g_watermask.size(), g_watermask.type());
		stream.enqueueDownload(g_watermask, temp);
		stream.waitForCompletion();
		minMaxLoc(temp, &mn, &mx);
		printf("masked:  min = %f, max = %f\n", mn, mx);
		//temp = nscale::PixelOperations::mod<int>(temp, 256);
		temp = (temp - mn) * (255.0 / (mx-mn));
		imwrite("test/out-gpu-watershed-oligoIII.1.png", temp);



		printf("watermask size: %d %d,  type %d\n", g_watermask.rows, g_watermask.cols, g_watermask.type());
		//	printf("g_border size: %d %d,  type %d\n", g_border.rows, g_border.cols, g_border.type());
		//	Mat watermask2(g_border.size(), g_border.type());
		//	stream.enqueueDownload(g_watermask, watermask2);
		//	stream.waitForCompletion();
		//	printf("here\n");

		g_watermask.release();
		g_distance2.release();
		g_seg_big.release();
		//	g_border.release();

		//	minMaxLoc(watermask2, &mn, &mx);
		//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));

		// to show the segmentation, use modulus to separate adjacent object's values
		//watermask2 = nscale::PixelOperations::mod(watermask2, 16) * 16;

		//	minMaxLoc(watermask2, &mn, &mx);
		//	printf("watershed:  min = %f, max = %f\n", mn, mx);
		//	watermask = nscale::PixelOperations::mod<int>(watermask2, 256);


		//	imwrite("test/out-gpu-watershed-dw-oligoIII.1.png", watermask);



		//	t1 = cci::common::event::timestampInUS();
		//	g_watermask = nscale::gpu::watershedCA(g_dummy, g_distance2, 8, stream);
		//	stream.waitForCompletion();
		//	t2 = cci::common::event::timestampInUS();
		//	std::cout << "gpu watershed CA loop took " << t2-t1 << "ms" << std::endl;
		//	g_watermask.download(watermask2);
		//	g_watermask.release();
		//
		//	minMaxLoc(watermask2, &mn, &mx);
		//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));
		//
		//	imwrite("test/out-gpu-watershed-ca-oligoIII.1.png", watermask2);
		//
		//
		//	// this would not work.  tested in matlab - over segment - it finds regions that are uniform.  nuclei are not uniform
		//	// compute the gradient mag
		//	// followed by hmin - this is input image
		//	// seed if inputimage with localmin, then labelled.
		//
		//	Mat gray(img.size(), CV_8U);
		//	cvtColor(img, gray, CV_BGR2GRAY);
		//	GpuMat g_gray(img.size(), CV_8U);
		//	stream.enqueueUpload(gray, g_gray);
		//    stream.waitForCompletion();
		//
		//	Mat gray_nuclei(nuclei.size(), gray.type());
		//	gray.copyTo(gray_nuclei, seg_big);
		//	GpuMat g_nuclei(gray_nuclei.size(), gray_nuclei.type());
		//	stream.enqueueUpload(gray_nuclei, g_nuclei);
		//
		//	GpuMat dx(g_nuclei.size(), CV_32F);
		//	Sobel(g_nuclei, dx, CV_32F, 0, 1);
		//	GpuMat dy(g_nuclei.size(), CV_32F);
		//	Sobel(g_nuclei, dy, CV_32F, 1, 0);
		//	GpuMat gradMag(g_nuclei.size(), CV_32F);
		//	magnitude(dx, dy, gradMag, stream);
		//	std::cout << "computed grad mag" << std::endl;
		//	dx.release();
		//	dy.release();
		//
		//	GpuMat hmin = nscale::gpu::imhmin(gradMag, 1.0f, 8, stream);
		//	gradMag.release();
		//	g_watermask = nscale::gpu::watershedCA(g_dummy, hmin, 8, stream);
		//	hmin.release();
		//	g_watermask.download(watermask2);
		//	g_watermask.release();
		//
		//	minMaxLoc(watermask2, &mn, &mx);
		//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));
		//
		//	imwrite("test/out-gpu-watershed-gradmag-oligoIII.1.png", watermask2);
		//	g_nuclei.release();
		//	gray_nuclei.release();
		//	gray.release();
		//	g_gray.release();

		//	watermask2.release();
#endif

		seg_big.release();
		img.release();
	}
	return 0;
}

