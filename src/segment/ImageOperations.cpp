/*
 * ImageOperations.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include "ImageOperations.h"
#include <limits.h>
#include <iostream>
#include "MorphologicOperations.h"
#include "highgui.h"

using namespace cv;

namespace nscale {

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat bwselect(Mat binaryImage, Mat seeds, int connectivity) {
	// only works for binary images.  ~I and MAX-I are the same....
	Mat marker = Mat::zeros(seeds.size(), seeds.type());
	binaryImage.copyTo(marker, seeds);

	marker = imreconstructBinary<uchar>(marker, binaryImage, connectivity);

	return marker & binaryImage;
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat imfill(Mat binaryImage, Mat seeds, int connectivity) {
	T mx = std::numeric_limits<T>::max();
	Mat mask = mx - binaryImage;  // validated

	Mat marker = Mat::zeros(mask.size(), mask.type());

	mask.copyTo(marker, seeds);

	marker = imreconstructBinary<uchar>(marker, mask, connectivity);

	return binaryImage | marker;
}

template Mat bwselect<uchar>(Mat binaryImage, Mat seeds, int connectivity);
template Mat imfill<uchar>(Mat binaryImage, Mat seeds, int connectivity);


}
