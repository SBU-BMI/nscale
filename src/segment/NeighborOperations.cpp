/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "NeighborOperations.h"
#include <limits>


namespace nscale {

using namespace cv;

// BORDER NOT BETTER THAN ERODE.  DO NOT USE
// require padded image.
template <typename T>
Mat NeighborOperations::border(Mat& img, T background, int connectivity) {

/*	// SPECIFIC FOR OPEN CV CPU WATERSHED
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

	Mat result(img.size(), img.type());
	T *ptr, *ptrm1, *res;

	
	for(int y=1; y< img.rows; y++){
		ptr = img.ptr<T>(y);
		ptrm1 = img.ptr<T>(y-1);

		res = result.ptr<T>(y);
		for (int x = 1; x < img.cols - 1; ++x) {
			if (ptrm1[x] == background &&
				(ptr[x-1] == background || ptr[x+1] == background)) {
				res[x] = background;
			} else {
				res[x] = ptr[x];
			}
		}
	}

	return result;
*/
	CV_Assert(0);
}

//TURN OFF
//template Mat NeighborOperations::border<int>(Mat&, int background, int connectivity);
}


