/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "SCIOHistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "NeighborOperations.h"
#include "PixelOperations.h"
#include "highgui.h"
#include "float.h"
#include "utils.h"
#include "ConnComponents.h"
#include "SCIOUtilsLogger.h"
#include "FileUtils.h"


namespace nscale {

using namespace cv;


int SCIOHistologicalEntities::segmentNuclei(const std::string& in, const std::string& out,
		int &compcount, int *&bbox,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

	long long t1, t2, t3;
	t1 = ::cciutils::event::timestampInUS();

	// parse the input string
	string suffix2;
	suffix2.assign(".tif");
	FileUtils futils(suffix2);
	string filename = futils.getFile(const_cast<std::string&>(in));
	// get the image name
	size_t pos = filename.rfind('.');
	if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", in.c_str());
	string prefix = filename.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", in.c_str());
	string ystr = prefix.substr(pos + 1);
	prefix = prefix.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", in.c_str());
	string xstr = prefix.substr(pos + 1);
	string imagename = prefix.substr(0, pos);
	int tilex = atoi(xstr.c_str());
	int tiley = atoi(ystr.c_str());

	// first split.
	Mat input = imread(in);
	t2 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("read"), t1, t2, std::string(), ::cciutils::event::FILE_I));

	if (!input.data) return ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;

	t1 = ::cciutils::event::timestampInUS();

	Mat output;
	int status = ::nscale::SCIOHistologicalEntities::segmentNuclei(input, output, compcount, bbox, NULL, NULL);
	input.release();

	t2 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(90, std::string("compute"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (iresHandler == NULL) printf("iresHandler is null why?\n");

	if (status == ::nscale::SCIOHistologicalEntities::SUCCESS) {
		//t1 = ::cciutils::event::timestampInUS();
		if (iresHandler != NULL) iresHandler->saveIntermediate(output, 100, imagename.c_str(), tilex, tiley, in.c_str());

		//t2 = ::cciutils::event::timestampInUS();
		//if (logsession != NULL) logsession->log(cciutils::event(100, std::string("write"), t1, t2, std::string(), ::cciutils::event::FILE_O));
	}
	output.release();

	return status;
}


int SCIOHistologicalEntities::segmentNuclei(const Mat& img, Mat& output,
		int &compcount, int *&bbox,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	// image in BGR format
	if (!img.data) 	return ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;

//	Mat img2(Size(1024,1024), img.type());
//	resize(img, img2, Size(1024,1024));
//	namedWindow("orig image", CV_WINDOW_AUTOSIZE);
//	imshow("orig image", img2);

	int status;

	long long t1, t2, t3;
	t1 = ::cciutils::event::timestampInUS();
	// first split.
	std::vector<Mat> bgr;
	split(img, bgr);
	t2 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(10, std::string("toRGB"), t1, t2, std::string(), ::cciutils::event::COMPUTE));

	// then check background
	t1 = ::cciutils::event::timestampInUS();
	std::vector<unsigned char> bgThresh;
	bgThresh.push_back(220);	
	bgThresh.push_back(220);	
	bgThresh.push_back(220);	
	Mat bg;
	float ratio = backgroundRatio(bgr, bgThresh, bg, logsession, iresHandler);
	bgThresh.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(bg, 20);
	bg.release();	
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(20, std::string("background"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(21, std::string("save background"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	if (ratio >= 0.99) {
		bgr[0].release();
		bgr[1].release();
		bgr[2].release();
		bgr.clear();
		return ::nscale::SCIOHistologicalEntities::BACKGROUND;
	} else if (ratio >= 0.9) {
		bgr[0].release();
		bgr[1].release();
		bgr[2].release();
		bgr.clear();
		return ::nscale::SCIOHistologicalEntities::BACKGROUND_LIKELY;
	}


	// next get RBC
	t1 = ::cciutils::event::timestampInUS();
	std::vector<double> rbcThresh;
	rbcThresh.push_back(5.0);
	rbcThresh.push_back(4.0);
	Mat rbc = getRBC(bgr, rbcThresh, logsession, iresHandler);
	rbcThresh.clear();	
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(rbc, 30);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(30, std::string("RBC"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(31, std::string("save RBC"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	// and separately find the gray scale candidates
	t1 = ::cciutils::event::timestampInUS();
	Mat grayNu;
	status = grayCandidates(grayNu, bgr[2], MORPH_ELLIPSE, 19, logsession, iresHandler);
	bgr[0].release();
	bgr[1].release();
	bgr[2].release();
	bgr.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(grayNu, 40);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(40, std::string("GrayNU"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(41, std::string("save GrayNU"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	if (status != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		grayNu.release();
		return status;	
	}

	// convert to binary
	t1 = ::cciutils::event::timestampInUS();
	std::vector<unsigned char> grayThresh;
	grayThresh.push_back(80);
	grayThresh.push_back(45);
	std::vector<int> sizeThresh1;
	sizeThresh1.push_back(11);
	sizeThresh1.push_back(1000);
	Mat binNu;
	status = grayToBinaryCandidate(binNu, grayNu, grayThresh, sizeThresh1, logsession, iresHandler);
	grayNu.release();
	grayThresh.clear();
	sizeThresh1.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(binNu, 50);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(50, std::string("NuMask"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(51, std::string("save NuMask"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	if (status != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		binNu.release();
		rbc.release();
		return status;	
	}


	// removeRBC
	t1 = ::cciutils::event::timestampInUS();
	std::vector<int> sizeThresh2;
	sizeThresh2.push_back(30);
	sizeThresh2.push_back(std::numeric_limits<int>::max());
	Mat seg_norbc;
	status = removeRBC(seg_norbc, binNu, rbc, sizeThresh2, logsession, iresHandler);
	binNu.release();
	rbc.release();
	sizeThresh2.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(seg_norbc, 60);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(60, std::string("removeRBC"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(61, std::string("save removeRBC"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	if (status != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		seg_norbc.release();
		return status;	
	}


	// separate Nu
	t1 = ::cciutils::event::timestampInUS();
	Mat seg_nonoverlap;
	status = separateNuclei(seg_nonoverlap, seg_norbc, img, 1.0, logsession, iresHandler);
	seg_norbc.release();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(seg_nonoverlap, 70);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(70, std::string("separateNuclei"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(71, std::string("save separateNuclei"), t2, t3, std::string(), ::cciutils::event::FILE_O));

	if (status != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		seg_nonoverlap.release();
		return status;	
	}

	// clean up
	t1 = ::cciutils::event::timestampInUS();
	std::vector<int> sizeThresh3;
	sizeThresh3.push_back(21);
	sizeThresh3.push_back(1000);
	status = finalCleanup(output, seg_nonoverlap, sizeThresh3, compcount, bbox, logsession, iresHandler);
	sizeThresh3.clear();
	seg_nonoverlap.release();
	t2 = ::cciutils::event::timestampInUS();
	if (iresHandler != NULL) iresHandler->saveIntermediate(output, 80);
	t3 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(80, std::string("finalCleanup"), t1, t2, std::string(), ::cciutils::event::COMPUTE));
	if (logsession != NULL) logsession->log(cciutils::event(81, std::string("save finalCleanup"), t2, t3, std::string(), ::cciutils::event::FILE_O));
	return status;

}


float SCIOHistologicalEntities::backgroundRatio(const std::vector<Mat>& bgr, const std::vector<unsigned char> &thresholds, Mat &bg, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

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
	bg = (bgr[0] > thresholds[0]) & (bgr[1] > thresholds[1]) & (bgr[2] > thresholds[2]);
	
	int bgArea = countNonZero(bg);
	
	return (float)bgArea / (float)(bgr[0].size().area());
}
	
Mat SCIOHistologicalEntities::getRBC(const std::vector<Mat>& bgr, const std::vector<double> &thresholds, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

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
	double T1 = thresholds[0];
	double T2 = thresholds[1];
	Size s = bgr[0].size();
	Mat bd(s, CV_32FC1);
	Mat gd(s, bd.type());
	Mat rd(s, bd.type());

	bgr[0].convertTo(bd, bd.type(), 1.0, FLT_EPSILON);
	bgr[1].convertTo(gd, gd.type(), 1.0, FLT_EPSILON);
	bgr[2].convertTo(rd, rd.type(), 1.0, 0.0);

	Mat imR2G = rd / gd;
	Mat imR2B = (rd / bd) > 1.0;
	bd.release();
	gd.release();
	rd.release();

	Mat bw1 = imR2G > T1;
	Mat bw2 = imR2G > T2;
	Mat rbc;
	if (countNonZero(bw1) > 0) {
		rbc = bwselect<unsigned char>(bw2, bw1, 8) & imR2B;
	} else {
		rbc = Mat::zeros(bw2.size(), bw2.type());
	}
	imR2G.release();
	imR2B.release();
	bw1.release();
	bw2.release();

	return rbc;
}
	
int SCIOHistologicalEntities::grayCandidates(Mat &diffIm, const Mat &r, const int kernelType, const int kernelRadius, ::cciutils::SCIOLogSession* logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {


	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	Mat rc = ::nscale::PixelOperations::invert<unsigned char>(r);

	Mat rc_open(rc.size(), rc.type());
	//Mat disk19 = getStructuringElement(MORPH_ELLIPSE, Size(19,19));
	// structuring element is not the same between matlab and opencv.  using the one from matlab explicitly....
	// (for 4, 6, and 8 connected, they are approximations).
	unsigned char disk19raw[361] = {
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
	rc_open = ::nscale::morphOpen<unsigned char>(rc, disk19);

	Mat rc_recon = ::nscale::imreconstruct<unsigned char>(rc_open, rc, 8);


	diffIm = rc - rc_recon;

	rc.release();
	rc_open.release();
	rc_recon.release();
	disk19vec.clear();
	disk19.release();

	return ::nscale::SCIOHistologicalEntities::CONTINUE;

}
	
int SCIOHistologicalEntities::grayToBinaryCandidate(Mat &seg_norbc, const Mat &diffIm, const std::vector<unsigned char> &colorThresholds, const std::vector<int> &sizeThresholds, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	Mat diffIm2 = diffIm > colorThresholds[0];

	Mat bw1 = ::nscale::imfillHoles<unsigned char>(diffIm2, true, 4);
	diffIm2.release();

	Mat bw1_t;
	int resultcode = areaThreshold(bw1_t, bw1, sizeThresholds, logsession, iresHandler);
	bw1.release();

	Mat bw2 = diffIm > colorThresholds[1];;

	/*
	 *
    [rows,cols] = ind2sub(size(diffIm),ind);
    seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc;
    seg_nohole = imfill(seg_norbc,'holes');
    seg_open = imopen(seg_nohole,strel('disk',1));
	 *
	 */

	seg_norbc = ::nscale::bwselect<unsigned char>(bw2, bw1_t, 8);
	bw1_t.release();
	bw2.release();

	return ::nscale::SCIOHistologicalEntities::CONTINUE;	

}
	
int SCIOHistologicalEntities::areaThreshold(Mat &result, const Mat &binary, const std::vector<int> &sizeThresholds, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

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
	int compcount2;

	result = ::nscale::bwareaopen2(binary, false, true, sizeThresholds[0], sizeThresholds[1], 8, compcount2);

	if (compcount2 == 0) {
		return ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	} else 
		return ::nscale::SCIOHistologicalEntities::CONTINUE;

}

int SCIOHistologicalEntities::removeRBC(Mat &seg_big, const Mat &seg_norbc, const Mat &rbc, const std::vector<int> &sizeThresholds, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

	Mat temp = seg_norbc & (rbc == 0);


	Mat seg_nohole = ::nscale::imfillHoles<unsigned char>(temp, true, 4);
	temp.release();

//	// can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
//	// because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	Mat seg_open = ::nscale::morphOpen<unsigned char>(seg_nohole, disk3);
		
	seg_nohole.release();
	disk3.release();	

	/*
	 *
	seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));
	 */
	
	int compcount2;
	seg_big = ::nscale::bwareaopen2(seg_open, false, true, sizeThresholds[0], sizeThresholds[1], 8, compcount2);
	seg_open.release();

	if (compcount2 == 0) {
		return ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	} else 
		return ::nscale::SCIOHistologicalEntities::CONTINUE;


}

int SCIOHistologicalEntities::separateNuclei(Mat &seg_nonoverlap, const Mat &seg_big_t, const Mat &img, const double h, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {


	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	Mat seg_big = Mat::zeros(seg_big_t.size(), seg_big_t.type());
	dilate(seg_big_t, seg_big, disk3);
	disk3.release();

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

	dist = - dist;  // appears to work better this way.

	// then set the background to -inf and do imhmin
	// appears to work better with -inf as background
	Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
	dist.copyTo(distance, seg_big);

	dist.release();



	// then do imhmin. (prevents small regions inside bigger regions)
	Mat distance2 = ::nscale::imhmin<float>(distance, h, 8);
	distance.release();

	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */

	Mat nuclei = Mat::zeros(img.size(), img.type());
	img.copyTo(nuclei, seg_big);

	// watershed in openCV requires labels.  input foreground > 0, 0 is background
	// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
	Mat watermask = ::nscale::watershed2(nuclei, distance2, 8);
	distance2.release();
	nuclei.release();

	seg_nonoverlap = Mat::zeros(seg_big.size(), seg_big.type());
	seg_big.copyTo(seg_nonoverlap, (watermask >= 0));
	seg_big.release();
	

	return ::nscale::SCIOHistologicalEntities::CONTINUE;

}
	
int SCIOHistologicalEntities::finalCleanup(Mat &seg, const Mat &seg_nonoverlap, const std::vector<int> &sizeThresholds, int &compcount, int *&bbox, ::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

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
	int compcount2;
	Mat seg_t = ::nscale::bwareaopen2(seg_nonoverlap, false, true, sizeThresholds[0], sizeThresholds[1], 4, compcount2);
	if (compcount2 == 0) {
		seg.release();
		return ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	}

	
	/*
	 *     %CHANGE
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);
	 *
	 */
	Mat final = ::nscale::imfillHoles<unsigned char>(seg_t, true, 8);
	seg_t.release();

	// MASK approach
	seg = nscale::bwlabel2(final, 8, true);
	final.release();

	::nscale::ConnComponents cc;
	bbox = cc.boundingBox(seg.cols, seg.rows, (int *)seg.data, 0, compcount);

	return ::nscale::SCIOHistologicalEntities::SUCCESS;


}










}
