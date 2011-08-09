/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "ScanlineOperations.h"
#include <algorithm>
#include <iterator>
#include <iostream>

namespace nscale {

using namespace cv;


bool ScanlineOperations::compareX (cv::Point i, cv::Point j)
{
  return (i.x < j.x);
}
bool ScanlineOperations::compareY (cv::Point i, cv::Point j)
{
  return (i.y < j.y);
}


/*
 * contour is a not approximated or compressed.  sort Points by x, then by y (assuming order preserving).
 *   then just walk down the list to count - lots of special cases...
 *
 */
uint64_t ScanlineOperations::getContourArea(const std::vector<cv::Point>& contour) {

	if (contour.size() < 3) return contour.size();

	// need to clean contour first, then duplicate.
	std::vector<cv::Point> newContour = duplicateVertices(cleanContour(contour));

	//std::cout << "cleaned boundary " << newContour << std::endl;

	// 2 sorts.  first by x, then by y.  the sort by y needs to preserve order
	stable_sort(newContour.begin(), newContour.end(), compareX);
	stable_sort(newContour.begin(), newContour.end(), compareY);

	//std::cout << "sorted boundary " << newContour << std::endl;


	// now determine the scanlines.
	std::vector<cv::Point>::iterator it = newContour.begin();
	std::vector<cv::Point>::iterator last = newContour.end();

	cv::Point start, end;

	uint64_t area = 0;

	// count even segment.  the start and end points should all be paired now...
	for ( ; it < last; ++it) {
		start = *it;  //std::cout << " start x = " << start.x << " start y = " << start.y << std::endl;
		end = *(++it); //std::cout << " end x = " << end.x << " end y = " << end.y << std::endl;
		CV_Assert(start.y == end.y);

		area += abs(end.x - start.x) + 1;
	}

	return area;
}

// this cleans up the contour to get it ready for fill or for counting area.
// assumes the boundary pixels are DENSE, and contour points are ORDERED on the boundary
std::vector<cv::Point> ScanlineOperations::cleanContour(const std::vector<cv::Point>& contour) {
	if (contour.size() < 3) return contour;

	// get the output storage.
	std::vector<cv::Point> newContour;
	newContour.reserve(contour.size());

	std::vector<cv::Point>::const_iterator it = contour.begin();
	std::vector<cv::Point>::const_iterator last = contour.end();

	// remove horizontal edges if any.
	cv::Point start, end, curr;
	start = *it;
	end = *it;
	newContour.push_back(start);
	int y = start.y;
	++it;

	for (; it < last; ++it) {
		curr = *it;

		// if y has not changed, then just move end forward.  the end node is added only when y changes
		// if y has changed, then add end if
		if (y != curr.y) {
			// not on the same line, so horizontal line has terminated.
			// now add the end, reset the iteration, and continue;
			if (start != end) newContour.push_back(end);  // only insert end if not same as start
			start = curr;
			newContour.push_back(start);
			y = start.y;
		}
		end = curr;  // keep the last entry in case need to insert into new queue
	}
	// note at the end, the last entry in the contour has been added to the new contour,
	// unless it's a horizontal edge.  so need to check to see if I need to add end.
	if (start != end) newContour.push_back(end);  // only insert end if not same as start

	// at then end of the list, the loop closes. but could have start segment and end segment
	// both part of a horizontal edge.  check and remove redundant points
	it = newContour.begin();
	y = it->y;
	std::vector<cv::Point>::reverse_iterator rit = newContour.rbegin();
	if (y == rit->y) {  // only process if start and end are on same line
		++it;
		++rit;

		// do the end first.
		// if end and it's predecessor are on same y, remove end
		if (rit->y == y) newContour.pop_back();

		// if the start node and its next node are on same scanline, remove start.
		if (it->y == y) newContour.erase(newContour.begin());

	}

	return newContour;

}

// this cleans up the contour to get it ready for fill or for counting area.
// assumes the boundary pixels are DENSE, and contour points are ORDERED on the boundary
std::vector<cv::Point> ScanlineOperations::duplicateVertices(const std::vector<cv::Point>& contour) {
	if (contour.size() < 3) return contour;

	// get the output storage.
	std::vector<cv::Point> newContour;
	newContour.reserve(contour.size() * 2);

	std::vector<cv::Point>::const_iterator it = contour.begin();
	std::vector<cv::Point>::const_iterator last = contour.end();

	// remove horizontal edges if any.
	cv::Point v1, v2;
	newContour.push_back(*it);
	int y0 = it->y;
	v1 = *(++it);
	newContour.push_back(v1);
	int y1 = v1.y;
	++it;

	int y2;
	for (; it < last; ++it) {
		v2 = *it;
		y2 = v2.y;

		// if vertex, then add another entry
		if (((y0 > y1) && (y2 > y1)) || ((y0 < y1) && (y2 < y1))) {
			newContour.push_back(v1);
		}
		newContour.push_back(v2);

		// set up the next iteration
		v1 = v2;
		y0 = y1;
		y1 = y2;

	}
	// note at the end, the last entry in the contour has been added to the new contour.

	// at then end of the list, the loop closes. now need to check to see if the last node
	// and the first nodes are vertices.
	it = newContour.begin();
	std::vector<cv::Point>::reverse_iterator rit = newContour.rbegin();

	// check the end
	v2 = *it;
	v1 = *rit;
	++rit;
	y0 = rit->y;
	y1 = v1.y;
	y2 = v2.y;

	if (((y0 > y1) && (y2 > y1)) || ((y0 < y1) && (y2 < y1))) {
		newContour.push_back(v1);
	}

	// check the beginning
	v1 = v2;
	y0 = y1;
	y1 = y2;
	v2 = *(++it);
	y2 = v2.y;
	if (((y0 > y1) && (y2 > y1)) || ((y0 < y1) && (y2 < y1))) {
		newContour.push_back(v1);
	}

	return newContour;

}


}


