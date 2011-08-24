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



/*
 * contour is a not approximated or compressed.  sort Points by x, then by y (assuming order preserving).
 *   then just walk down the list to count - lots of special cases...
 *
 ** note that findContours return cw for foreground, and ccw for background.
 */
uint64_t ScanlineOperations::getContourArea(const std::vector<cv::Point>& contour, bool foreground) {

	if (contour.size() < 3) return contour.size();

	// need to clean contour first, then duplicate.
	std::vector<cv::Point> newContour = duplicateVertices(reduceHorizontalEdges(contour), foreground);

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

/*
 * contour is a not approximated or compressed.  sort Points by x, then by y (assuming order preserving).
 *   then just walk down the list to count - lots of special cases...
 * note that findContours return cw for foreground, and ccw for background.
 *
 * TODO:  still not right....
 */
uint64_t ScanlineOperations::getContourArea(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx) {

	CV_Assert(idx > -1);

	int size = contours[idx].size();

	if (size < 3) return size;

	// need to clean contour first, then duplicate.
	std::vector<cv::Point> newContour;


	//if (size < 30) std::cout << "orig boundary " << contours[idx] << std::endl;
	std::vector<cv::Point> currContour = duplicateVertices(reduceHorizontalEdges(contours[idx]), true);
	//if (size < 30) std::cout << "cleaned boundary " << currContour << std::endl;
	newContour.insert(newContour.end(), currContour.begin(), currContour.end());

	// now get the holes
	int i = hierarchy[idx][2];
	for ( ; i >= 0; i = hierarchy[i][0]) {
		//if (size < 30) std::cout << "orig hole boundary " << contours[i] << std::endl;
		currContour = duplicateVertices(reduceHorizontalEdges(contours[i]), false);
		//if (size < 30) std::cout << "cleaned hole boundary " << currContour << std::endl;
		newContour.insert(newContour.end(), currContour.begin(), currContour.end());
	}

//	if (size < 30) std::cout << "cleaned all boundary " << newContour << std::endl;

	// 2 sorts.  first by x, then by y.  the sort by y needs to preserve order
	stable_sort(newContour.begin(), newContour.end(), compareX);
	stable_sort(newContour.begin(), newContour.end(), compareY);

//	if (size < 30) std::cout << "sorted all boundary " << newContour << std::endl;


	// now determine the scanlines.
	std::vector<cv::Point>::iterator it = newContour.begin();
	std::vector<cv::Point>::iterator last = newContour.end();

	cv::Point start, end;

	uint64_t area = 0;

	// count even segment.  the start and end points should all be paired now...
	for ( ; it < last; ++it) {
		start = *it;  //std::cout << " start x = " << start.x << " start y = " << start.y << std::endl;
		++it;
		end = *it; //std::cout << " end x = " << end.x << " end y = " << end.y << std::endl;
		CV_Assert(start.y == end.y);

		area += end.x - start.x + 1;
	}

	return area;
}

// this cleans up the contour to get it ready for fill or for counting area.
// assumes the boundary pixels are DENSE, and contour points are ORDERED on the boundary.
std::vector<cv::Point> ScanlineOperations::reduceHorizontalEdges(const std::vector<cv::Point>& contour) {
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
		// do the end first.
		// if end and it's predecessor are on same y, remove end
		++rit;
		if (rit->y == y) newContour.pop_back();

		// if the start node and its next node are on same scanline, remove start.
		++it;
		if (it->y == y) newContour.erase(newContour.begin());

	}

	return newContour;

}


void ScanlineOperations::duplicateVertices1(const int& dx0, const int& dy0, const int& dx1, const int& dy1,
	const cv::Point& v1, std::vector<cv::Point>& newContour, bool cw) {

	// only check when going from not horizontal to horizontal, or vice versa.
	// no colinear horizontal.
	if (dy0 == 0) {   // leaving horizontal edge
		// only add if concave.
		// cross product is dx0dy1-dx1dy0;  dy0 is 0, so just need to check dx0dy1 < 0.
		if (!(sameSign(dx0, dy1) ^ cw)) {   // "not the same sign (dx0dy1 < 0) when ccw", "same sign (dx0dy1 >= 0) with cw".
			newContour.pop_back();  // remove inner vertex
//			std::cout << " 1.1. removed. because of " << dx0 << " and " << dy1 << " have same sign" << std::endl;
		}
	} else if (dy1 == 0) {
		// only add if concave.
		// cross product is dx0dy1-dx1dy0;  dy1 is 0, so just need to check dx1dy0 > 0.
		if (sameSign(dx1, dy0) ^ cw) {  // same sign (dx1dy0 > 0) when ccw.
			newContour.pop_back();  // remove inner vertex
//			std::cout << " 1.2. removed. because of " << dx1 << " and " << dy0 << " have same sign" << std::endl;
		}
	} else if (! sameSign(dy0, dy1)) { // if 2 segment are on the same side of vertex, then dy0 and dy1 are diff in sign. doesn't matter about ccw
//		std::cout << " at point. " << dy0 << " and " << dy1 << std::endl;
		// neither segments are horizontal
		if (sameSign(dy0, dx0) ^ cw) {
			// first segment is going to upper right (lower left), and the next segment is going to lower right (upper left) = convex
			newContour.pop_back(); // add the inner vertex
//			std::cout << " 1.3. removed. because of " << dx0 << " and " << dy0 << " have same sign" << std::endl;

		} else {
			// the frist segment is going upper left (lower right), second is lower left (upper right) = convex
			newContour.push_back(v1);  // add extra external vertex
//			std::cout << " 1.4. inserted at " << v1.x << " and " << v1.y << std::endl;
		}
	}

}


// this cleans up the contour to get it ready for fill or for counting area.
// assumes the boundary pixels are DENSE, and contour points are ORDERED on the boundary
std::vector<cv::Point> ScanlineOperations::duplicateVertices(const std::vector<cv::Point>& contour, bool foreground) {
	if (contour.size() < 3) return contour;

	// get the output storage.
	std::vector<cv::Point> newContour;
	newContour.reserve(contour.size() * 2);

	std::vector<cv::Point>::const_iterator it = contour.begin();
	std::vector<cv::Point>::const_iterator last  = contour.end();

	// save the first and second entries, and compute the x and y deltas.
	cv::Point v1, first, second;
	first = *it;
	newContour.push_back(*it);
	int dy0 = it->y;
	int dx0 = it->x;
	++it;
	second = *it;
	newContour.push_back(*it);
	int y1 = it->y;
	int x1 = it->x;
	++it;

	dy0 = y1 - dy0;
	dx0 = x1 - dx0;

	int dy1, dx1, y2, x2;
	int cross;
	for (; it < last; ++it) {

		y2 = it->y;
		x2 = it->x;
		dy1 = y2 - y1;
		dx1 = x2 - x1;

		duplicateVertices1(dx0, dy0, dx1, dy1, v1, newContour, foreground);
		v1 = *it;
		newContour.push_back(v1);

		// set up the next iteration
		dy0 = dy1;
		dx0 = dx1;
		y1 = y2;
		x1 = x2;
	}
	// note at the end, the last entry in the contour has been added to the new contour.

	// at then end of the list, the loop closes. now need to check to see if the last node
	// and the first nodes are vertices.

	// check the end of list
	y2 = first.y;
	x2 = first.x;
	dy1 = y2 - y1;
	dx1 = x2 - x1;

	duplicateVertices1(dx0, dy0, dx1, dy1, v1, newContour, foreground);

	// set up the next iteration
	dy0 = dy1;
	dx0 = dx1;
	v1 = first;
	y1 = y2;
	x1 = x2;

	// check the beginning of list
	y2 = second.y;
	x2 = second.x;
	dy1 = y2 - y1;
	dx1 = x2 - x1;

	// need to do this separately - removal is from the front...
	// only check when going from not horizontal to horizontal, or vice versa.
	// no colinear horizontal.
	if (dy0 == 0) {   // leaving horizontal edge
		// only add if concave.
		// cross product is dx0dy1-dx1dy0;  dy0 is 0, so just need to check dx0dy1 < 0.
		if (!(sameSign(dx0, dy1) ^ foreground)) {   // "not the same sign (dx0dy1 < 0) when ccw", "same sign (dx0dy1 >= 0) with cw".
			newContour.erase(newContour.begin());  // remove inner vertex from front of list
//			std::cout << " 1. removed head. because of " << dx0 << " and " << dy1 << " have same sign" << std::endl;
		}
	} else if (dy1 == 0) {
		// only add if concave.
		// cross product is dx0dy1-dx1dy0;  dy1 is 0, so just need to check dx1dy0 > 0.
		if (sameSign(dx1, dy0) ^ foreground) {  // same sign (dx1dy0 > 0) when ccw.
			newContour.erase(newContour.begin());  // remove inner vertex
//			std::cout << " 2. removed head. because of " << dx1 << " and " << dy0 << " have same sign" << std::endl;
		}
	} else if (! sameSign(dy0, dy1)) { // if 2 segment are on the same side of vertex, then dy0 and dy1 are diff in sign. doesn't matter about ccw
		// neither segments are horizontal
		if (sameSign(dy0, dx0) ^ foreground) {
			// first segment is going to upper right (lower left), and the next segment is going to lower right (upper left) = concave
			newContour.erase(newContour.begin());  // remove the inner vertex
//			std::cout << " 3. removed head. because of " << dx0 << " and " << dy0 << " have same sign" << std::endl;
		} else {
			// the frist segment is going upper left (lower right), second is lower left (upper right) = convex
			newContour.push_back(v1);  // add extra external vertex
//			std::cout << " 4. inserted at " << v1.x << " and " << v1.y << std::endl;

		}
	}

	return newContour;

}


}


