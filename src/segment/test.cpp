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
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include <time.h>
#include "utils.h"

namespace {

using ::cv;

bool areaThreshold1(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx) {
	return nscale::contourAreaFilter(contours, hierarchy, idx, 11, 1000);
}


int main (int argc, char **argv){
/*	// allow walk through of the directory
	const char* impath = argc > 1 ? argv[1];
	// get the files - from http://ubuntuforums.org/showthread.php?t=1409202
	vector<string> files();
	Dir *dir;
	struct dirent *dp;
	if ((dir = std::opendir(impath.c_str())) == NULL) {
		std::cout << "ERROR(" << errno << ") opening" << impath << std::endl;
		return errno;
	}
	while ((dp = readdir(dir)) != NULL) {
		files.push_back(string(dp->d_name));
		if ()
	}
	closedir(dir);


	// set the output path
	const char* resultpath = argc > 2 ? argv[2];
*/
	const char* imagename = argc > 1 ? argv[1] : "lena.jpg";

	// need to go through filesystem

	Mat img = imread(imagename);

	if (!img.data) return -1;


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
	Mat gray(img.size(), CV_8UC1);
	cvtColor(img, gray, CV_BGR2GRAY);

	std::vector<Mat> bgr;
	split(img, bgr);
	Mat background = (bgr[0] > 220) & (bgr[1] > 220) & (bgr[2] > 220);
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

	imwrite("/home/tcpan/PhD/path/rbc.pbm", rbc);

	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	Mat rc = std::numeric_limits<uchar>::max() - bgr[2];
	Mat rc_open(rc.size(), rc.type());
	Mat disk21 = getStructuringElement(MORPH_ELLIPSE, Size(21,21));
	morphologyEx(rc, rc_open, CV_MOP_OPEN, disk21, Point(-1, -1), 1);
	Mat rc_recon = nscale::imreconstruct<uchar>(rc_open, rc, 8);
	Mat diffIm = rc - rc_recon;

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	uchar G1 = 80;
	Mat diffIm2 = diffIm > G1;
	Mat bw1 = nscale::imfillHoles<uchar>(diffIm2, true, 8);

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
	Mat bw1 = nscale::bwareaopen<uchar>(bw1, areaThreshold1, std::numeric_limits<uchar>::max(), 8);
	if (countNonZero(bw1) == 0) return;

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
	Mat seg_nohole = nscale::imfillHoles<uchar>(seg_norbc, true, 8);
	Mat seg_open = Mat::zeros(seg_nohole.size(), seg_nohole.type());
	Mat disk3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, Point(-1, -1), 1);

	/*
	 *
	seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));

    distance = -bwdist(~seg_big);
    distance(~seg_big) = -Inf;
    distance2 = imhmin(distance, 1);
	 *
	 */
	// bwareaopen is done as a area threshold.
	Mat seg_big_t = nscale::bwareaopen<uchar>(seg_open, 30, std::numeric_limits<uchar>::max(), 8);
	dilate(seg_big_t, seg_big, disk3);
	// distance transform:  matlab code is doing this:
	// invert the image so nuclei candidates are holes
	// compute the distance (distance of nuclei pixels to background)
	// negate the distance.  so now background is still 0, but nuclei pixels have negative distances
	// set background to -inf
	Mat dist(seg_big.size(), CV_32FC1);
	distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	dist = 0.0 - dist;
	Mat distance(dist.size(), dist.type());
	distance = std::numeric_limits<float>::min();
	dist.copyTo(distance, seg_big);
	// then do imhmin.
	Mat distance2 = nscale::imhmin<float>(distance, 1.0f, 8);



	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */
	// watershed in openCV requires labels.
	Mat watermask = nscale::watershed2(distance2, 8);
	Mat seg_nonoverlap = Mat::zeros(seg_big.size(), seg_big.type());
	seg_big.copyTo(seg_nonoverlap, watermask);

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
	if (countNonZero(seg) == 0) return;


	/*
	 *     %CHANGE
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);
	 *
	 */
	// don't worry about bwlabel.
	Mat output = nscale::imfillHoles<uchar>(seg, true, 8);


//	waitKey();

	return 0;
}

}
