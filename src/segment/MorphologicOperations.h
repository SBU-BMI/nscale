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
cv::Mat imreconstruct(const cv::Mat& seeds, const cv::Mat& image, int connectivity);

template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, const cv::Mat& binaryImage, int connectivity);


template <typename T>
cv::Mat imfill(const cv::Mat& image, const cv::Mat& seeds, bool binary, int connectivity);

template <typename T>
cv::Mat imfillHoles(const cv::Mat& image, bool binary, int connectivity);



template <typename T>
cv::Mat bwselect(const cv::Mat& binaryImage, const cv::Mat& seeds, int connectivity);

cv::Mat_<int> bwlabel(const cv::Mat& binaryImage, int connectivity);
// incorporates a filter for the contours.
template <typename T>
cv::Mat bwlabelFiltered(const cv::Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<cv::Point> >&, const std::vector<cv::Vec4i>&, int), int connectivity);

// inclusive min, exclusive max
bool contourAreaFilter(const std::vector<std::vector<cv::Point> >& contours, const std::vector<cv::Vec4i>& hierarchy, int idx, int minArea, int maxArea);

// inclusive min, exclusive max.
template <typename T>
cv::Mat bwareaopen(const cv::Mat& binaryImage, int minSize, int maxSize, int connectivity);

template <typename T>
cv::Mat imhmin(const cv::Mat& image, T h, int connectivity);

cv::Mat watershed2(const cv::Mat& image, int connectivity);

template <typename T>
cv::Mat localMaxima(const cv::Mat& image, int connectivity);
template <typename T>
cv::Mat localMinima(const cv::Mat& image, int connectivity);


}


#endif /* MORPHOLOGICOPERATION_H_ */
