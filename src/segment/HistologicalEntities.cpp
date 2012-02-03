/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "HistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "highgui.h"
#include "float.h"
#include "utils.h"

namespace nscale {

using namespace cv;



Mat HistologicalEntities::getRBC(const Mat& img,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	CV_Assert(img.channels() == 3);

	std::vector<Mat> bgr;
	split(img, bgr);
	return getRBC(bgr, logger, iresHandler);
}

Mat HistologicalEntities::getRBC(const std::vector<Mat>& bgr,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
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
	std::cout.precision(5);
	double T1 = 5.0;
	double T2 = 4.0;
	Size s = bgr[0].size();
	Mat bd(s, CV_32FC1);
	Mat gd(s, bd.type());
	Mat rd(s, bd.type());

	bgr[0].convertTo(bd, bd.type(), 1.0, FLT_EPSILON);
	bgr[1].convertTo(gd, gd.type(), 1.0, FLT_EPSILON);
	bgr[2].convertTo(rd, rd.type(), 1.0, 0.0);

	Mat imR2G = rd / gd;
	Mat imR2B = (rd / bd) > 1.0;

	if (iresHandler) iresHandler->saveIntermediate(imR2G, 101);
	if (iresHandler) iresHandler->saveIntermediate(imR2B, 102);


	Mat bw1 = imR2G > T1;
	Mat bw2 = imR2G > T2;
	Mat rbc;
	if (countNonZero(bw1) > 0) {
//		imwrite("test/in-bwselect-marker.pgm", bw2);
//		imwrite("test/in-bwselect-mask.pgm", bw1);
		rbc = bwselect<uchar>(bw2, bw1, 8) & imR2B;
	} else {
		rbc = Mat::zeros(bw2.size(), bw2.type());
	}

	return rbc;
}

Mat HistologicalEntities::getBackground(const Mat& img,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	CV_Assert(img.channels() == 3);

	std::vector<Mat> bgr;
	split(img, bgr);
	return getBackground(bgr, logger, iresHandler);
}

Mat HistologicalEntities::getBackground(const std::vector<Mat>& bgr,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
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


	return (bgr[0] > 220) & (bgr[1] > 220) & (bgr[2] > 220);
}



int HistologicalEntities::segmentNuclei(const std::string& in, const std::string& out,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	Mat input = imread(in);
	if (!input.data) return ::nscale::HistologicalEntities::INVALID_IMAGE;

	Mat output(input.size(), CV_8U, Scalar(0));

	int status = ::nscale::HistologicalEntities::segmentNuclei(input, output, logger, iresHandler);

	if (status == ::nscale::HistologicalEntities::SUCCESS)
		imwrite(out, output);

	return status;
}


// S1
int HistologicalEntities::plFindNucleusCandidates(const Mat& img, Mat& seg_norbc,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	std::vector<Mat> bgr;
	split(img, bgr);
	if (logger) logger->logTimeSinceLastLog("toRGB");

	Mat background = ::nscale::HistologicalEntities::getBackground(bgr, logger, iresHandler);

	int bgArea = countNonZero(background);
	float ratio = (float)bgArea / (float)(img.size().area());
//TODO: TMEP	std::cout << " background size: " << bgArea << " ratio: " << ratio << std::endl;
	if (logger) logger->log("backgroundRatio", ratio);

	if (ratio >= 0.99) {
		//TODO: TEMP std::cout << "background.  next." << std::endl;
		if (logger) logger->logTimeSinceLastLog("background");
		return ::nscale::HistologicalEntities::BACKGROUND;
	} else if (ratio >= 0.9) {
		//TODO: TEMP std::cout << "background.  next." << std::endl;
		if (logger) logger->logTimeSinceLastLog("background likely");
		return ::nscale::HistologicalEntities::BACKGROUND_LIKELY;
	}

	if (logger) logger->logTimeSinceLastLog("background");
	if (iresHandler) iresHandler->saveIntermediate(background, 1);

	Mat rbc = ::nscale::HistologicalEntities::getRBC(bgr, logger, iresHandler);
	if (logger) logger->logTimeSinceLastLog("RBC");
	int rbcPixelCount = countNonZero(rbc);
	if (logger) logger->log("RBCPixCount", rbcPixelCount);

//	imwrite("test/out-rbc.pbm", rbc);
	if (iresHandler) iresHandler->saveIntermediate(rbc, 2);

	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	Mat rc = ::nscale::PixelOperations::invert<uchar>(bgr[2]);
	if (logger) logger->logTimeSinceLastLog("invert");

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
	rc_open = ::nscale::morphOpen<unsigned char>(rc, disk19);
//	morphologyEx(rc, rc_open, CV_MOP_OPEN, disk19, Point(-1, -1), 1);
	if (logger) logger->logTimeSinceLastLog("open19");
	if (iresHandler) iresHandler->saveIntermediate(rc_open, 3);

// for generating test data 
//	imwrite("test/in-imrecon-gray-marker.pgm", rc_open);
//	imwrite("test/in-imrecon-gray-mask.pgm", rc);
//	exit(0);
// END for generating test data
	Mat rc_recon = ::nscale::imreconstruct<uchar>(rc_open, rc, 8);
	if (iresHandler) iresHandler->saveIntermediate(rc_recon, 4);


	Mat diffIm = rc - rc_recon;
//	imwrite("test/out-redchannelvalleys.ppm", diffIm);
	if (logger) logger->logTimeSinceLastLog("reconToNuclei");
	int rc_openPixelCount = countNonZero(rc_open);
	if (logger) logger->log("rc_openPixCount", rc_openPixelCount);
	if (iresHandler) iresHandler->saveIntermediate(diffIm, 5);

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	uchar G1 = 80;
	Mat diffIm2 = diffIm > G1;
	if (logger) logger->logTimeSinceLastLog("threshold1");
	if (iresHandler) iresHandler->saveIntermediate(diffIm2, 6);

	Mat bw1 = ::nscale::imfillHoles<uchar>(diffIm2, true, 4);
//	imwrite("test/out-rcvalleysfilledholes.ppm", bw1);
	if (logger) logger->logTimeSinceLastLog("fillHoles1");
	if (iresHandler) iresHandler->saveIntermediate(bw1, 7);

//	// TODO: change back
//	return ::nscale::HistologicalEntities::SUCCESS;
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
	bw1 = ::nscale::bwareaopen<uchar>(bw1, 11, 1000, 8);
	if (iresHandler) iresHandler->saveIntermediate(bw1, 8);
	if (countNonZero(bw1) == 0) {
		if (logger) logger->logTimeSinceLastLog("areaThreshold1");
		return ::nscale::HistologicalEntities::NO_CANDIDATES_LEFT;
	}
//	imwrite("test/out-nucleicandidatessized.ppm", bw1);
	if (logger) logger->logTimeSinceLastLog("areaThreshold1");


	uchar G2 = 45;
	Mat bw2 = diffIm > G2;
	if (iresHandler) iresHandler->saveIntermediate(bw2, 9);


	/*
	 *
    [rows,cols] = ind2sub(size(diffIm),ind);
    seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc;
    seg_nohole = imfill(seg_norbc,'holes');
    seg_open = imopen(seg_nohole,strel('disk',1));
	 *
	 */

	seg_norbc = ::nscale::bwselect<uchar>(bw2, bw1, 8);
	if (iresHandler) iresHandler->saveIntermediate(seg_norbc, 10);

	seg_norbc = seg_norbc & (rbc == 0);
	if (iresHandler) iresHandler->saveIntermediate(seg_norbc, 11);

//	imwrite("test/out-nucleicandidatesnorbc.ppm", seg_norbc);
	if (logger) logger->logTimeSinceLastLog("blobsGt45");


	return ::nscale::HistologicalEntities::CONTINUE;

}


// A4
int HistologicalEntities::plSeparateNuclei(const Mat& img, const Mat& seg_open, Mat& seg_nonoverlap,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	/*
	 *
	seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));
	 */
	// bwareaopen is done as a area threshold.
	Mat seg_big_t = ::nscale::bwareaopen<uchar>(seg_open, 30, std::numeric_limits<int>::max(), 8);
	if (logger) logger->logTimeSinceLastLog("30To1000");
	if (iresHandler) iresHandler->saveIntermediate(seg_big_t, 14);


	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));

	Mat seg_big = Mat::zeros(seg_big_t.size(), seg_big_t.type());
	dilate(seg_big_t, seg_big, disk3);
	if (iresHandler) iresHandler->saveIntermediate(seg_big, 15);

//	imwrite("test/out-nucleicandidatesbig.ppm", seg_big);
	if (logger) logger->logTimeSinceLastLog("dilate");

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

	// opencv: compute the distance to nearest zero
	// matlab: compute the distance to the nearest non-zero
	distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	double mmin, mmax;
	minMaxLoc(dist, &mmin, &mmax);
	if (iresHandler) iresHandler->saveIntermediate(dist, 16);

	// invert and shift (make sure it's still positive)
	//dist = (mmax + 1.0) - dist;
	dist = - dist;  // appears to work better this way.

//	cciutils::cv::imwriteRaw("test/out-dist", dist);

	// then set the background to -inf and do imhmin
	//Mat distance = Mat::zeros(dist.size(), dist.type());
	// appears to work better with -inf as background
	Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
	dist.copyTo(distance, seg_big);
//	cciutils::cv::imwriteRaw("test/out-distance", distance);
	if (logger) logger->logTimeSinceLastLog("distTransform");
	if (iresHandler) iresHandler->saveIntermediate(distance, 17);



	// then do imhmin. (prevents small regions inside bigger regions)
	Mat distance2 = ::nscale::imhmin<float>(distance, 1.0f, 8);
	if (logger) logger->logTimeSinceLastLog("imhmin");
	if (iresHandler) iresHandler->saveIntermediate(distance2, 18);


//cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);


	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */

	Mat nuclei = Mat::zeros(img.size(), img.type());
//	Mat distance3 = distance2 + (mmax + 1.0);
//	Mat dist4 = Mat::zeros(distance3.size(), distance3.type());
//	distance3.copyTo(dist4, seg_big);
//	Mat dist5(dist4.size(), CV_8U);
//	dist4.convertTo(dist5, CV_8U, (std::numeric_limits<uchar>::max() / mmax));
//	cvtColor(dist5, nuclei, CV_GRAY2BGR);
	img.copyTo(nuclei, seg_big);
	if (logger) logger->logTimeSinceLastLog("nucleiCopy");
	if (iresHandler) iresHandler->saveIntermediate(nuclei, 19);

	// watershed in openCV requires labels.  input foreground > 0, 0 is background
	// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
	Mat watermask = ::nscale::watershed2(nuclei, distance2, 8);
//	cciutils::cv::imwriteRaw("test/out-watershed", watermask);
	if (logger) logger->logTimeSinceLastLog("watershed");
	if (iresHandler) iresHandler->saveIntermediate(watermask, 20);


	seg_nonoverlap = Mat::zeros(seg_big.size(), seg_big.type());
	seg_big.copyTo(seg_nonoverlap, (watermask >= 0));
	// erode a bit
	if (logger) logger->logTimeSinceLastLog("water to mask");
	if (iresHandler) iresHandler->saveIntermediate(seg_nonoverlap, 21);


	return ::nscale::HistologicalEntities::CONTINUE;

}




int HistologicalEntities::segmentNuclei(const Mat& img, Mat& output,
		::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	// image in BGR format
	if (!img.data) return ::nscale::HistologicalEntities::INVALID_IMAGE;

//	Mat img2(Size(1024,1024), img.type());
//	resize(img, img2, Size(1024,1024));
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", img2);

	if (logger) logger->logT0("start");
	if (iresHandler) iresHandler->saveIntermediate(img, 0);

	Mat seg_norbc;
	int findCandidateResult = ::nscale::HistologicalEntities::plFindNucleusCandidates(img, seg_norbc, logger, iresHandler);
	if (findCandidateResult != ::nscale::HistologicalEntities::CONTINUE) {
		return findCandidateResult;
	}


	Mat seg_nohole = ::nscale::imfillHoles<uchar>(seg_norbc, true, 4);
	if (logger) logger->logTimeSinceLastLog("fillHoles2");
	if (iresHandler) iresHandler->saveIntermediate(seg_nohole, 12);

	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	Mat seg_open = ::nscale::morphOpen<unsigned char>(seg_nohole, disk3);
//	// can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
//	// because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
//	//	morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, Point(1,1)); //, Point(-1, -1), 1, BORDER_REFLECT);
//	Mat t_seg_nohole;
//	copyMakeBorder(seg_nohole, t_seg_nohole, 1, 1, 1, 1, BORDER_CONSTANT, std::numeric_limits<uchar>::max());
//	Mat t_seg_erode = Mat::zeros(t_seg_nohole.size(), t_seg_nohole.type());
//	erode(t_seg_nohole, t_seg_erode, disk3);
//	Mat seg_erode = t_seg_erode(Rect(1, 1, seg_nohole.cols, seg_nohole.rows));
//	Mat t_seg_erode2;
//	copyMakeBorder(seg_erode,t_seg_erode2, 1, 1, 1, 1, BORDER_CONSTANT, std::numeric_limits<uchar>::min());
//	Mat t_seg_open = Mat::zeros(t_seg_erode2.size(), t_seg_erode2.type());
//	dilate(t_seg_erode2, t_seg_open, disk3);
//	Mat seg_open = t_seg_open(Rect(1,1,seg_nohole.cols, seg_nohole.rows));
//	t_seg_open.release();
//	t_seg_erode2.release();
//	seg_erode.release();
//	t_seg_erode.release();
//	t_seg_nohole.release();

//	imwrite("test/out-nucleicandidatesopened.ppm", seg_open);
	if (logger) logger->logTimeSinceLastLog("openBlobs");
	if (iresHandler) iresHandler->saveIntermediate(seg_open, 13);


	Mat seg_nonoverlap;
	int sepResult = ::nscale::HistologicalEntities::plSeparateNuclei(img, seg_open, seg_nonoverlap, logger, iresHandler);
	if (sepResult != ::nscale::HistologicalEntities::CONTINUE) {
		return sepResult;
	}

	Mat twm;
	copyMakeBorder(seg_nonoverlap, twm, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(std::numeric_limits<uchar>::max()));
	Mat t_nonoverlap = Mat::zeros(twm.size(), twm.type());
	erode(twm, t_nonoverlap, disk3);
	seg_nonoverlap = t_nonoverlap(Rect(1,1,seg_nonoverlap.cols, seg_nonoverlap.rows));
//	imwrite("test/out-seg_nonoverlap.ppm", seg_nonoverlap);
	if (logger) logger->logTimeSinceLastLog("watershed erode");
	if (iresHandler) iresHandler->saveIntermediate(seg_nonoverlap, 22);


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
	Mat seg = ::nscale::bwareaopen<uchar>(seg_nonoverlap, 21, 1000, 4);
	if (logger) logger->logTimeSinceLastLog("20To1000");
	if (countNonZero(seg) == 0) {
		return ::nscale::HistologicalEntities::NO_CANDIDATES_LEFT;
	}
	if (iresHandler) iresHandler->saveIntermediate(seg, 23);

//	imwrite("test/out-seg.ppm", seg);


	/*
	 *     %CHANGE
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);
	 *
	 */
	// don't worry about bwlabel.
	output = ::nscale::imfillHoles<uchar>(seg, true, 8);
	if (logger) logger->logTimeSinceLastLog("fillHolesLast");

//	imwrite("test/out-nuclei.ppm", seg);

//	if (logger) logger->endSession();

	return ::nscale::HistologicalEntities::SUCCESS;

}


}
