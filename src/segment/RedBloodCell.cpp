/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "RedBloodCell.h"
#include <iostream>
#include "highgui.h"
#include "float.h"
#include "utils.h"

namespace nscale {

Mat RedBloodCell::rbcMask(Mat img) {
	std::vector<Mat> bgr;
	split(img, bgr);
	return rbcMask(bgr);
}

Mat RedBloodCell::rbcMask(std::vector<Mat> bgr) {
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
	Mat bw1 = imR2G > T1;
	Mat bw2 = imR2G > T2;
	Mat bw3 = ~bw2;

	// multiple seeds.  need to get the ids and then iterate through
	Mat rbc = Mat::zeros(s, CV_8UC1);
	uchar * rowPointer;
	if (countNonZero(bw1) > 0) {
		// iterate over all pixels

		uint64_t t1 = cciutils::ClockGetTime();
		// internals of bwselect
		for (int j = 0; j < bw1.rows; j++) {
			rowPointer = bw1.ptr<uchar>(j);
			for (int i = 0; i < bw1.cols; i++) {
				if (rowPointer[i] == 0) continue;
				// comparison operation produces 0 and 255.  need to fill with 255.
				bw3 = floodFill(bw3, Point(i, j), 255, NULL, Scalar(), Scalar(), 8);
			}
		}
		// internals of bwselect
		Mat bw4 = bw2 & bw3;
		uint64_t t2 = cciutils::ClockGetTime();
		std::cout << " bwselect took " << t2 - t1 << "ms" << std::endl;

		// now the rest
		rbc = bw4 & ((rd / bd) > 1.0);
	}

	return rbc;
}

}
