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

	template <typename T>
	static ::cv::Mat border(const ::cv::Mat& img, T background, int connectivity);

};

namespace gpu {
class NeighborOperations {

public:

	template <typename T>
	static ::cv::gpu::GpuMat border(const ::cv::gpu::GpuMat& img, T background, int connectivity, ::cv::gpu::Stream& stream);
};

}

}

#endif /* NEIGHBOROPERATIONS_H_ */
