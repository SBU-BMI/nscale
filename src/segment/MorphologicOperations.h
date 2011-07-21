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
cv::Mat imreconstruct(const cv::Mat& seeds, const cv::Mat& image, int connectivity = 8);

template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, const cv::Mat& image, int connectivity = 8);


template <typename T>
cv::Mat bwselectBinary(cv::Mat binaryImage, cv::Mat seeds, int connectivity);

//template <typename T>
//cv::Mat bwlabel(cv::Mat binaryImage, int connectivity);


template <typename T>
cv::Mat imfillBinary(cv::Mat binaryImage, cv::Mat seeds, int connectivity);

template <typename T>
cv::Mat imfillHoles(cv::Mat binaryImage, int connectivity);
template <typename T>
cv::Mat imfillHolesBinary(cv::Mat image, int connectivity);


}

#endif /* MORPHOLOGICOPERATION_H_ */
