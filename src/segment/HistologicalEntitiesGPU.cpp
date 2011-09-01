/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */
#define HAVE_CUDA 1

#include "HistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "highgui.h"
#include "float.h"
#include "utils.h"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

using namespace cv;

namespace gpu {

using namespace cv::gpu;



GpuMat HistologicalEntities::getRBC(const std::vector<GpuMat>& bgr, Stream& stream) {
	CV_Assert(bgr.size() >= 3);
	/*
	%T1=2.5; T2=2;
    T1=5; T2=4;

	imR2G = double(r)./(double(g)+eps);
    bw1 = imR2G > T1;
    bw2 = imR2G > T2;
    ind = find(bw1);
    if ~isempty(ind)
        [rows, cols]=ind2sub(size(imR2G),ind);
        rbc = bwselect(bw2,cols,rows,8) & (double(r)./(double(b)+eps)>1);
    else
        rbc = zeros(size(imR2G));
    end
	 */
	float T1 = 5.0;
	float T2 = 4.0;
	Size s = bgr[0].size();
	int newType = CV_32FC1;
	GpuMat bd, gd, rd;

	stream.enqueueConvert(bgr[0], bd, newType, 1.0, FLT_EPSILON);
	stream.enqueueConvert(bgr[1], gd, newType, 1.0, FLT_EPSILON);
	stream.enqueueConvert(bgr[2], rd, newType);
	stream.waitForCompletion();

	GpuMat imR2G, imR2B;
	divide(rd, gd, imR2G, stream);
	divide(rd, bd, imR2B, stream);
	stream.waitForCompletion();
	rd.release();
	gd.release();
	bd.release();
	GpuMat bw3 = PixelOperations::threshold<float>(imR2B, 1.0, std::numeric_limits<float>::max(), stream);
	GpuMat bw1 = PixelOperations::threshold<float>(imR2G, T1, std::numeric_limits<float>::max(), stream);
	GpuMat bw2 = PixelOperations::threshold<float>(imR2G, T2, std::numeric_limits<float>::max(), stream);
	stream.waitForCompletion();
	imR2G.release();
	imR2B.release();

	GpuMat rbc(s, CV_8UC1);
	stream.enqueueMemSet(rbc, Scalar(0));
	if (countNonZero(bw1) > 0) {

		GpuMat temp = nscale::gpu::bwselect<unsigned char>(bw2, bw1, 8, stream);
		bitwise_and(temp, bw3, rbc, GpuMat(), stream);
		stream.waitForCompletion();
		temp.release();
	}
	bw1.release();
	bw2.release();
	bw3.release();

	return rbc;
}

GpuMat HistologicalEntities::getBackground(const std::vector<GpuMat>& g_bgr, Stream& stream) {

	GpuMat b1, g1, r1;
	GpuMat g_bg(g_bgr[0].size(), CV_8U);
	stream.enqueueMemSet(g_bg, Scalar(0));
	uchar max = std::numeric_limits<uchar>::max();
	threshold(g_bgr[0], b1, 220, max, THRESH_BINARY, stream);
	threshold(g_bgr[1], g1, 220, max, THRESH_BINARY, stream);
	threshold(g_bgr[2], r1, 220, max, THRESH_BINARY, stream);
	bitwise_and(b1, r1, g_bg, g1, stream);
	stream.waitForCompletion();

	b1.release();
	g1.release();
	r1.release();

	return g_bg;
}


int HistologicalEntities::segmentNuclei(const Mat& img, Mat& output) {
	// image in BGR format

//	Mat img2(Size(1024,1024), img.type());
//	resize(img, img2, Size(1024,1024));
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", img2);

	/*
	* this part to decide if the tile is background or foreground
	THR = 0.9;
    grayI = rgb2gray(I);
    area_bg = length(find(I(:,:,1)>=220&I(:,:,2)>=220&I(:,:,3)>=220));
    ratio = area_bg/numel(grayI);
    if ratio >= THR
        return;
    end
	 */
	GpuMat g_img;
	Stream stream;
	stream.enqueueUpload(img, g_img);

	std::vector<GpuMat> g_bgr;
	split(g_img, g_bgr, stream);

	GpuMat g_bg = getBackground(g_bgr, stream);
	int bgArea = countNonZero(g_bg);
	stream.waitForCompletion();
	g_bg.release();

	float ratio = (float)bgArea / (float)(img.size().area());
	std::cout << " background size: " << bgArea << " ratio: " << ratio << std::endl;
	if (ratio >= 0.9) {
		std::cout << "background.  next." << std::endl;
		return -1;
	}

	uint64_t t1 = cciutils::ClockGetTime();
	GpuMat g_rbc = nscale::gpu::HistologicalEntities::getRBC(g_bgr, stream);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "rbc took " << t2-t1 << "ms" << std::endl;
	GpuMat g_r(g_bgr[2]);
	stream.waitForCompletion();
	g_bgr[0].release();
	g_bgr[1].release();
	g_bgr[2].release();


	Mat rbc(g_rbc.size(), g_rbc.type());
	stream.enqueueDownload(g_rbc, rbc);
	stream.waitForCompletion();
	imwrite("test/out-rbc.pbm", rbc);
	g_rbc.release();

	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	GpuMat g_rc = nscale::gpu::PixelOperations::invert<unsigned char>(g_r, stream);
	stream.waitForCompletion();
	g_r.release();
	GpuMat g_rc_open(g_rc.size(), g_rc.type());
	//Mat disk19 = getStructuringElement(MORPH_ELLIPSE, Size(19,19));
	// structuring element is not the same between matlab and opencv.  using the one from matlab explicitly....
	// (for 4, 6, and 8 connected, they are approximations).
	uchar disk19raw[361] = {
			0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
			0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
			0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
	std::vector<unsigned char> disk19vec(disk19raw, disk19raw+361);
	Mat disk19(disk19vec);
	disk19 = disk19.reshape(1, 19);
	imwrite("test/out-rcopen-strel.pbm", disk19);
	// filter doesnot check borders.  so need to create border.
	GpuMat rc_border;
	copyMakeBorder(g_rc, rc_border, 9,9,9,9, Scalar(0), stream);
	stream.waitForCompletion();
	GpuMat rc_roi(rc_border, Range(9, 9+ g_rc.size().height), Range(9, 9+g_rc.size().width));
	morphologyEx(rc_roi, g_rc_open, MORPH_OPEN, disk19, Point(-1, -1), 1, stream);
	Mat rc_open(g_rc_open.size(), g_rc_open.type());
	stream.enqueueDownload(g_rc_open, rc_open);
	stream.waitForCompletion();
	rc_roi.release();
	rc_border.release();
	imwrite("test/out-rcopen.ppm", rc_open);


	GpuMat g_rc_recon = nscale::gpu::imreconstruct<unsigned char>(g_rc_open, g_rc, 8, stream);
//	GpuMat g_rc_recon = nscale::gpu::imreconstruct2<unsigned char>(g_rc_open, g_rc, 8, stream);
	GpuMat g_diffIm;
	subtract(g_rc, g_rc_recon, g_diffIm, stream);
	stream.waitForCompletion();
	g_rc_open.release();
	g_rc.release();
	g_rc_recon.release();

	Mat diffIm(g_diffIm.size(), g_diffIm.type());
	stream.enqueueDownload(g_diffIm, diffIm);
	stream.waitForCompletion();
	imwrite("test/out-redchannelvalleys.ppm", diffIm);

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	uchar G1 = 80;
	GpuMat g_diffIm2;
	threshold(g_diffIm, g_diffIm2, G1, std::numeric_limits<unsigned char>::max(), THRESH_BINARY, stream);
	stream.waitForCompletion();
	g_diffIm.release();

	GpuMat g_bw1 = nscale::gpu::imfillHoles<unsigned char>(g_diffIm2, true, 4, stream);
	Mat bw1(g_bw1.size(), g_bw1.type());
	stream.enqueueDownload(g_bw1, bw1);
	stream.waitForCompletion();
	g_diffIm2.release();
	g_bw1.release();
	imwrite("test/out-rcvalleysfilledholes.ppm", bw1);

	g_img.release();
	std::cout << "Completed GPU phase" << std::endl;
/*
 *     %CHANGE
    [L] = bwlabel(bw1, 8);
    stats = regionprops(L, 'Area');
    areas = [stats.Area];

    %CHANGE
    ind = find(areas>10 & areas<1000);
    bw1 = ismember(L,ind);
    bw2 = diffIm>G2;
    ind = find(bw1);

    if isempty(ind)
        return;
    end
 *
 */
	bw1 = nscale::bwareaopen<uchar>(bw1, 11, 1000, 8);
	if (countNonZero(bw1) == 0) return -1;
	imwrite("test/out-nucleicandidatessized.ppm", bw1);



	uchar G2 = 45;
	Mat bw2 = diffIm > G2;



	/*
	 *
    [rows,cols] = ind2sub(size(diffIm),ind);
    seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc;
    seg_nohole = imfill(seg_norbc,'holes');
    seg_open = imopen(seg_nohole,strel('disk',1));
	 *
	 */

	Mat seg_norbc = nscale::bwselect<uchar>(bw2, bw1, 8);
	seg_norbc = seg_norbc & (rbc == 0);
	imwrite("test/out-nucleicandidatesnorbc.ppm", seg_norbc);
	Mat seg_nohole = nscale::imfillHoles<uchar>(seg_norbc, true, 4);
	Mat seg_open = Mat::zeros(seg_nohole.size(), seg_nohole.type());
	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, Point(-1, -1), 1);
	imwrite("test/out-nucleicandidatesopened.ppm", seg_open);

	/*
	 *
	seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));
	 */
	// bwareaopen is done as a area threshold.
	Mat seg_big_t = nscale::bwareaopen<uchar>(seg_open, 30, std::numeric_limits<int>::max(), 8);
	Mat seg_big = Mat::zeros(seg_big_t.size(), seg_big_t.type());
	dilate(seg_big_t, seg_big, disk3);
	imwrite("test/out-nucleicandidatesbig.ppm", seg_big);

	/*
	 *
		distance = -bwdist(~seg_big);
		distance(~seg_big) = -Inf;
		distance2 = imhmin(distance, 1);
		 *
	 *
	 */
	// distance transform:  matlab code is doing this:
	// invert the image so nuclei candidates are holes
	// compute the distance (distance of nuclei pixels to background)
	// negate the distance.  so now background is still 0, but nuclei pixels have negative distances
	// set background to -inf

	// really just want the distance map.  CV computes distance to 0.
	// background is 0 in output.
	// then invert to create basins
	Mat dist(seg_big.size(), CV_32FC1);

	distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	double mmin, mmax;
	minMaxLoc(dist, &mmin, &mmax);
	dist = (mmax + 1.0) - dist;
	cciutils::cv::imwriteRaw("test/out-dist", dist);

	// then set the background to -inf and do imhmin
	Mat distance = Mat::zeros(dist.size(), dist.type());
	dist.copyTo(distance, seg_big);
	cciutils::cv::imwriteRaw("test/out-distance", distance);
	// then do imhmin.
	//Mat distance2 = nscale::imhmin<float>(distance, 1.0f, 8);
	//cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);


	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */

	Mat nuclei = Mat::zeros(img.size(), img.type());
	img.copyTo(nuclei, seg_big);

	// watershed in openCV requires labels.  input foreground > 0, 0 is background
	Mat watermask = nscale::watershed2(nuclei, distance, 8);
	cciutils::cv::imwriteRaw("test/out-watershed", watermask);


	Mat seg_nonoverlap = Mat::zeros(seg_big.size(), seg_big.type());
	seg_big.copyTo(seg_nonoverlap, (watermask > 0));
	imwrite("test/out-seg_nonoverlap.ppm", seg_nonoverlap);

	/*
     %CHANGE
    [L] = bwlabel(seg_nonoverlap, 4);
    stats = regionprops(L, 'Area');
    areas = [stats.Area];

    %CHANGE
    ind = find(areas>20 & areas<1000);

    if isempty(ind)
        return;
    end
    seg = ismember(L,ind);
	 *
	 */
	Mat seg = nscale::bwareaopen<uchar>(seg_nonoverlap, 21, 1000, 4);
	if (countNonZero(seg) == 0) return -1;
	imwrite("test/out-seg.ppm", seg);


	/*
	 *     %CHANGE
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);
	 *
	 */
	// don't worry about bwlabel.
	output = nscale::imfillHoles<uchar>(seg, true, 8);
	imwrite("test/out-nuclei.ppm", seg);



	return 0;
}

}}
