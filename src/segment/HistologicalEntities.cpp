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

namespace nscale {

using namespace cv;


Mat HistologicalEntities::getRBC(const Mat& img) {
	CV_Assert(img.channels() == 3);

	std::vector<Mat> bgr;
	split(img, bgr);
	return getRBC(bgr);
}

Mat HistologicalEntities::getRBC(const std::vector<Mat>& bgr) {
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
	Mat bd(s, CV_64FC1);
	Mat gd(s, bd.type());
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
		rbc = bwselectBinary<uchar>(bw2, bw1, 8) & imR2B;
	} else {
		rbc = Mat::zeros(bw2.size(), bw2.type());
	}

	return rbc;
}

}
