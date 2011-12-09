/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "PixelOperations.h"
#include <limits>


namespace nscale {

using namespace cv;

template <typename T>
Mat PixelOperations::invert(const Mat& img) {
	// write the raw image
	CV_Assert(img.channels() == 1);

	if (std::numeric_limits<T>::is_integer) {

		if (std::numeric_limits<T>::is_signed) {
			Mat output;
			bitwise_not(img, output);
			return output + 1;
		} else {
			// unsigned int
			return std::numeric_limits<T>::max() - img;
		}

	} else {
		// floating point type
		return -img;
	}


}

template <typename T>
Mat PixelOperations::mod(const Mat& img, T mod) {
	// write the raw image
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

	Mat result(img.size(), img.type());
	bitwise_and(img, Scalar(mod - (T)1), result);

	return result;
}

template Mat PixelOperations::invert<unsigned char>(const Mat&);
template Mat PixelOperations::invert<float>(const Mat&);

template Mat PixelOperations::mod<unsigned char>(const Mat&, unsigned char mod);
template Mat PixelOperations::mod<int>(const Mat&, int mod);

}


