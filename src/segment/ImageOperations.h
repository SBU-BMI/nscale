/*
 * ImageOperations.h
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#ifndef IMAGEOPERATIONS_H_
#define IMAGEOPERATIONS_H_
#include "cv.h"


namespace nscale {

template <typename T>
cv::Mat bwselect(cv::Mat binaryImage, cv::Mat seeds, int connectivity);

template <typename T>
cv::Mat imfill(cv::Mat binaryImage, cv::Mat seeds, int connectivity);


}

#endif /* IMAGEOPERATIONS_H_ */
