/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "ScanlineOperations.h"

namespace nscale {

using namespace cv;


/*
 * contour is a not approximated or compressed.  sort Points by x, then by y (assuming order preserving).
 *   then just walk down the list to count - lots of special cases...
 *
 */
uint64_t ScanlineOperations::getContourArea(const std::vector<cv::Point>& contour) {

}

// this cleans up the contour to get it ready for fill or for counting area.
// assumes all pixels on boundary are present, and contour points are ordered on the boundary
std::vector<cv::Point> ScanlineOperations::cleanContour(const std::vector<cv::Point>& contour) {
	// get the output storage.
	std::vector<cv::Point> newContour;
	newContour.reserve(contour.size());

	iterator<cv::Point> it = contour.begin();
	iterator<cv::Point> end = contour.end();

	// first remove horizontal edges if any
	cv::Point *start, *end, *curr;
	start = it;
	end = it;
	++it;

	for (; it < end; ++it) {
		curr = it;

		// if adjacent
		if (end.y == curr.y) {

		} else {
			// not on the same line, so reset the iteration, and also add the curr point to the new list
			start = curr;
			end = curr;
			newContour.push_back(curr);
		}

		}

	}


}


cv::Mat ScanlineOperations::fillContours(const std::vector<std::vector<cv::Point> >& contours, cv::Scalar& color) {

}

}


