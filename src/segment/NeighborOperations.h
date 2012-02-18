/*
 * ScanlineOperations.h
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#ifndef NEIGHBOROPERATIONS_H_
#define NEIGHBOROPERATIONS_H_

#include "cv.h"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class NeighborOperations {

public:

	// IMPLE DOES NOT WORK BETTER THAN ERODE.  
	// fixes watershed borders.
	template <typename T>
	static ::cv::Mat border(::cv::Mat& img, T background, int connectivity);

};

namespace gpu {
class NeighborOperations {

public:

	// fixes watershed borders
	template <typename T>
	static ::cv::gpu::GpuMat border(const ::cv::gpu::GpuMat& img, T background, int connectivity, ::cv::gpu::Stream& stream);
};

}

}

#endif /* NEIGHBOROPERATIONS_H_ */
