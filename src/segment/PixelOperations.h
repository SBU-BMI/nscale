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

	template <typename T>
	static ::cv::Mat mod(const ::cv::Mat& img, T mod);

	static void ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);
	static ::cv::Mat bgr2gray(const ::cv::Mat& img);
};

namespace gpu {
class PixelOperations {

public:

	template <typename T>
	static ::cv::gpu::GpuMat invert(const ::cv::gpu::GpuMat& img, ::cv::gpu::Stream& stream);

	template <typename T>
	static ::cv::gpu::GpuMat threshold(const ::cv::gpu::GpuMat& img, T lower, T upper, ::cv::gpu::Stream& stream);
<<<<<<< HEAD

	template <typename T>
	static ::cv::gpu::GpuMat mod(const ::cv::gpu::GpuMat& img, T mod, ::cv::gpu::Stream& stream);
=======
	static void convertIntToChar(GpuMat& input, GpuMat&result, Stream& stream);
	static void convertIntToCharAndRemoveBorder(GpuMat& input, GpuMat&result, int top, int bottom, int left, int right, Stream& stream);
	static void ColorDeconv( GpuMat& image, const Mat& M, const Mat& b, GpuMat& H, GpuMat& E, Stream& stream, bool BGR2RGB=true);
	static GpuMat bgr2gray(const GpuMat& img, Stream& stream);
>>>>>>> 44ee712be5a854c479743eb367a09c589810597a
};

}

}

#endif /* PIXELOPERATIONS_H_ */
