/*
 * ScanlineOperations.h
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#ifndef PIXELOPERATIONS_H_
#define PIXELOPERATIONS_H_

#include "cv.h"
#include "opencv2/gpu/gpu.hpp"
#include "utils.h"

using namespace cv;
using namespace cv::gpu;
using namespace std;

namespace nscale {

class PixelOperations {

public:

	template <typename T>
	static ::cv::Mat invert(const ::cv::Mat& img);
	void ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);
};

namespace gpu {
class PixelOperations {

public:

	template <typename T>
	static ::cv::gpu::GpuMat invert(const ::cv::gpu::GpuMat& img, ::cv::gpu::Stream& stream);

	template <typename T>
	static ::cv::gpu::GpuMat threshold(const ::cv::gpu::GpuMat& img, T lower, T upper, ::cv::gpu::Stream& stream);

	static void convertIntToChar(GpuMat& input, GpuMat&result, Stream& stream);

	static void convertIntToCharAndRemoveBorder(GpuMat& input, GpuMat&result, int top, int bottom, int left, int right, Stream& stream);
};

}

}

#endif /* PIXELOPERATIONS_H_ */
