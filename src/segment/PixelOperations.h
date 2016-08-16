/*
 * ScanlineOperations.h
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#ifndef PIXELOPERATIONS_H_
#define PIXELOPERATIONS_H_

#include "opencv2/opencv.hpp"

#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif 

#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define DllExport __declspec(dllexport)
#else
#define DllExport //nothing 
#endif


using namespace cv;
#ifdef WITH_CUDA
using namespace cv::gpu;
#endif
using namespace std;


namespace nscale {



	class DllExport PixelOperations {
public:

	template <typename T>
	static ::cv::Mat invert(const ::cv::Mat& img);

	template <typename T>
	static ::cv::Mat mod(::cv::Mat& img, T mod);

	//static void ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);
	static ::cv::Mat ComputeInverseStainMatrix(const Mat& M, const Mat& b);
	static ::std::vector<float> ComputeLookupTable();

	static void ColorDeconv(const Mat& image, const Mat& Q, const vector<float>& lut, Mat& H, Mat& E, bool BGR2RGB = true);
	static ::cv::Mat bgr2gray(const ::cv::Mat& img);

	template <typename T>
	static ::cv::Mat replace(const ::cv::Mat &img, T oldval, T newval);
};

#ifdef WITH_CUDA
namespace gpu {
	class DllExport PixelOperations {

public:

	template <typename T>
	static ::cv::gpu::GpuMat invert(const ::cv::gpu::GpuMat& img, ::cv::gpu::Stream& stream);

	template <typename T>
	static ::cv::gpu::GpuMat threshold(const ::cv::gpu::GpuMat& img, T lower, bool lower_inclusive, T upper, bool up_inclusive, ::cv::gpu::Stream& stream);
	template <typename T>
	static ::cv::gpu::GpuMat divide(const ::cv::gpu::GpuMat& num, const ::cv::gpu::GpuMat& den, ::cv::gpu::Stream& stream);
	template <typename T>
	static ::cv::gpu::GpuMat mask(const ::cv::gpu::GpuMat& input, const ::cv::gpu::GpuMat& mask, T background, ::cv::gpu::Stream& stream);

	template <typename T>
	static ::cv::gpu::GpuMat replace(const ::cv::gpu::GpuMat &img, T oldval, T newval, ::cv::gpu::Stream& stream);

	template <typename T>
	static ::cv::gpu::GpuMat mod(::cv::gpu::GpuMat& img, T mod, ::cv::gpu::Stream& stream);
	static void convertIntToChar(GpuMat& input, GpuMat&result, Stream& stream);
	static void convertIntToCharAndRemoveBorder(GpuMat& input, GpuMat&result, int top, int bottom, int left, int right, Stream& stream);
	static void ColorDeconv( GpuMat& image, const Mat& M, const Mat& b, GpuMat& H, GpuMat& E, Stream& stream, bool BGR2RGB=true);
	static GpuMat bgr2gray(const GpuMat& img, Stream& stream);

//	static void copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value, Stream& stream);
};

}
#endif

}

#endif /* PIXELOPERATIONS_H_ */
