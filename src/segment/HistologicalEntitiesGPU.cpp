/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "HistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "highgui.h"
#include "float.h"
#include "utils.h"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

namespace gpu {

using namespace cv;
using namespace cv::gpu;



GpuMat HistologicalEntities::getRBC(const std::vector<GpuMat>& bgr) {
	CV_Assert(bgr.size() == 3);
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
	double T1 = 5.0;
	double T2 = 4.0;
	Size s = bgr[0].size();
	GpuMat bd(s, CV_64FC1);
	GpuMat gd(s, bd.type());
	Mat rd(s, bd.type());

	bgr[0].convertTo(bd, bd.type(), 1.0, DBL_EPSILON);
	bgr[1].convertTo(gd, gd.type(), 1.0, DBL_EPSILON);
	bgr[2].convertTo(rd, rd.type(), 1.0, 0.0);

	Mat imR2G = rd / gd;
	Mat imR2B = (rd / bd) > 1.0;
	Mat bw1 = imR2G > T1;
	Mat bw2 = imR2G > T2;
	Mat rbc;
	if (countNonZero(bw1) > 0) {
		rbc = bwselect<uchar>(bw2, bw1, 8) & imR2B;
	} else {
		rbc = Mat::zeros(bw2.size(), bw2.type());
	}

	return rbc;
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

	GpuMat g_bg, b1, g1, r1;
	threshold(bgr[0], b1, 220, 255, THRESH_BINARY, stream);
	threshold(bgr[1], g1, 220, 255, THRESH_BINARY, stream);
	threshold(bgr[2], r1, 220, 255, THRESH_BINARY, stream);
	bitwise_and(b1, g1, g_bg, r1, stream);

	b1.release();
	g1.release();
	r1.release();

	Mat background;
	stream.enqueueDownload(g_bg, background);
	stream.waitForCompletion();
	int bgArea = countNonZero(background);
	float ratio = (float)bgArea / (float)(img.size().area());
	if (ratio >= 0.9) {
		std::cout << "background.  next." << std::endl;
		return -1;
	}

	uint64_t t1 = cciutils::ClockGetTime();
	Mat rbc = nscale::HistologicalEntities::getRBC(bgr);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "rbc took " << t2-t1 << "ms" << std::endl;

	imwrite("test/out-rbc.pbm", rbc);

	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	Mat rc = cciutils::cv::invert<uchar>(bgr[2]);
	Mat rc_open(rc.size(), rc.type());
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
	std::vector<uchar> disk19vec(disk19raw, disk19raw+361);
	Mat disk19(disk19vec);
	disk19 = disk19.reshape(1, 19);
	imwrite("test/out-rcopen-strel.pbm", disk19);
	morphologyEx(rc, rc_open, CV_MOP_OPEN, disk19, Point(-1, -1), 1);
	imwrite("test/out-rcopen.ppm", rc_open);

	Mat rc_recon = nscale::imreconstruct<uchar>(rc_open, rc, 8);
	Mat diffIm = rc - rc_recon;
	imwrite("test/out-redchannelvalleys.ppm", diffIm);

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	uchar G1 = 80;
	Mat diffIm2 = diffIm > G1;
	Mat bw1 = nscale::imfillHoles<uchar>(diffIm2, true, 4);
	imwrite("test/out-rcvalleysfilledholes.ppm", bw1);

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

}

}
