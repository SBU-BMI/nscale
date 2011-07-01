/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "RedBloodCell.h"

namespace nscale {
RedBloodCell::RedBloodCell() {
	// TODO Auto-generated constructor stub

}

RedBloodCell::~RedBloodCell() {
	// TODO Auto-generated destructor stub
}

cv::Mat RedBloodCell::rbcMask(cv::Mat img) {
	std::vector<cv::Mat> rgb;
	cv::split(img, rgb);
	return rbcMask(rgb);
}

cv::Mat RedBloodCell::rbcMask(std::vector<cv::Mat> rgb) {
	/*
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
	cv::Mat imR2G(rgb[0].size(), CV_64FC1);
	cv::Mat temp(rgb[0].size(), imR2G.type());
	double eps = 2.2204E-16;
	rgb[1].convertTo(temp, temp.type(), 1.0, eps);
	rgb[0].convertTo(imR2G, imR2G.type(), 1.0, 0.0);

	imR2G = imR2G / temp;

	cv::Mat imR2G(rgb[0].size(), CV_64FC1);
	cv::Mat temp(rgb[0].size(), CV_64FC1);

}

}
