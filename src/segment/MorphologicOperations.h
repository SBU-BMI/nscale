/*
 * MorphologicOperation.h
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#ifndef MORPHOLOGICOPERATION_H_
#define MORPHOLOGICOPERATION_H_

#include "cv.h"


namespace nscale {
// DOES NOT WORK WITH MULTICHANNEL.
template <typename T>
cv::Mat_<T> imreconstruct(const cv::Mat_<T>& seeds, const cv::Mat_<T>& image, int conn);

template <typename T>
cv::Mat_<T> imreconstructScan(const cv::Mat_<T>& seeds, const cv::Mat_<T>& image, int conn);

template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, const cv::Mat& image, int conn);


}

#endif /* MORPHOLOGICOPERATION_H_ */
