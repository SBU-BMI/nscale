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

template <typename T>
cv::Mat_<T> imreconstruct(const cv::Mat_<T>& image, const cv::Mat_<T>& seeds, int conn);

}

#endif /* MORPHOLOGICOPERATION_H_ */
