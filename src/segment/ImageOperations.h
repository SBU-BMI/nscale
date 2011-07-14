/*
 * ImageOperations.h
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#ifndef IMAGEOPERATIONS_H_
#define IMAGEOPERATIONS_H_
#include "cv.h"

using namespace cv;

namespace nscale {

class ImageOperations {
public:
	static Mat bwselect(Mat input, Mat seeds, int connectivity);
	static Mat imfill(Mat input, Mat seeds, int connectivity);

};

}

#endif /* IMAGEOPERATIONS_H_ */
