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


template <typename T>
Mat NeighborOperations::border(const Mat& img, T background, int connectivity) {
	// write the raw image
	printf("Border() is not implemented for CPU yet");
	CV_Assert(0);
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

	//Mat result(img.size(), img.type());

	// border processing

	// TODO: implement this to mirror GPU version.



	return img;
}

template Mat NeighborOperations::border<int>(const Mat&, int background, int connectivity);

}


