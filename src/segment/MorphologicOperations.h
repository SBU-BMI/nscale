/*
 * MorphologicOperation.h
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#ifndef MORPHOLOGICOPERATION_H_
#define MORPHOLOGICOPERATION_H_

#include "cv.hpp"
#include "opencv2/gpu/gpu.hpp"




namespace nscale {

// DOES NOT WORK WITH MULTICHANNEL.
template <typename T>
cv::Mat imreconstruct(const cv::Mat& seeds, const cv::Mat& image, int connectivity);

// downhill version...
cv::Mat imreconstructUChar(const cv::Mat& seeds, const cv::Mat& image, int connectivity);

template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, const cv::Mat& binaryImage, int connectivity);


template <typename T>
cv::Mat imfill(const cv::Mat& image, const cv::Mat& seeds, bool binary, int connectivity);

template <typename T>
cv::Mat imfillHoles(const cv::Mat& image, bool binary, int connectivity);



template <typename T>
cv::Mat bwselect(const cv::Mat& binaryImage, const cv::Mat& seeds, int connectivity);

cv::Mat_<int> bwlabel(const cv::Mat& binaryImage, bool contourOnly, int connectivity);
// incorporates a filter for the contours.
template <typename T>
cv::Mat bwlabelFiltered(const cv::Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<cv::Point> >&, const std::vector<cv::Vec4i>&, int),
		bool contourOnly, int connectivity);

// inclusive min, exclusive max
bool contourAreaFilter(const std::vector<std::vector<cv::Point> >& contours, const std::vector<cv::Vec4i>& hierarchy, int idx, int minArea, int maxArea);

// inclusive min, exclusive max.
template <typename T>
cv::Mat bwareaopen(const cv::Mat& binaryImage, int minSize, int maxSize, int connectivity);

template <typename T>
cv::Mat imhmin(const cv::Mat& image, T h, int connectivity);

cv::Mat_<int> watershed2(const cv::Mat& origImage, const cv::Mat_<float>& image, int connectivity);

template <typename T>
cv::Mat_<uchar> localMaxima(const cv::Mat& image, int connectivity);
template <typename T>
cv::Mat_<uchar> localMinima(const cv::Mat& image, int connectivity);


namespace gpu {
// GPU versions of the same functions.

// DOES NOT WORK WITH MULTICHANNEL.

template <typename T>
cv::gpu::GpuMat imreconstruct(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& image, int connectivity, cv::gpu::Stream& stream, unsigned int& iter);
template <typename T>
cv::gpu::GpuMat imreconstruct(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& image, int connectivity, cv::gpu::Stream& stream) {
	unsigned int iter;
	return imreconstruct<T>(seeds, image, connectivity, stream, iter);
};

template <typename T>
cv::gpu::GpuMat imreconstructBinary(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& binaryImage, int connectivity, cv::gpu::Stream& stream, unsigned int& iter);

template <typename T>
cv::gpu::GpuMat imreconstructBinary(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& binaryImage, int connectivity, cv::gpu::Stream& stream){
	unsigned int iter;
	return imreconstructBinary<T>(seeds, binaryImage,connectivity, stream, iter);
};

template <typename T>
cv::gpu::GpuMat imreconstructQ(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& image, int connectivity, cv::gpu::Stream& stream, unsigned int& iter);

template <typename T>
cv::gpu::GpuMat imreconstructQ(const cv::gpu::GpuMat& seeds, const cv::gpu::GpuMat& image, int connectivity, cv::gpu::Stream& stream) {
	unsigned int iter;
	return imreconstructQ<T>(seeds, image, connectivity, stream, iter);
};


//template <typename T>
//cv::Mat imfill(const cv::Mat& image, const cv::Mat& seeds, bool binary, int connectivity);
//
template <typename T>
cv::gpu::GpuMat imfillHoles(const cv::gpu::GpuMat& image, bool binary, int connectivity, cv::gpu::Stream& stream);



template <typename T>
cv::gpu::GpuMat bwselect(const cv::gpu::GpuMat& binaryImage, const cv::gpu::GpuMat& seeds, int connectivity, cv::gpu::Stream& stream);

//cv::Mat_<int> bwlabel(const cv::Mat& binaryImage, bool contourOnly, int connectivity);
//// incorporates a filter for the contours.
//template <typename T>
//cv::Mat bwlabelFiltered(const cv::Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<cv::Point> >&, const std::vector<cv::Vec4i>&, int),
//		bool contourOnly, int connectivity);
//
//// inclusive min, exclusive max
//bool contourAreaFilter(const std::vector<std::vector<cv::Point> >& contours, const std::vector<cv::Vec4i>& hierarchy, int idx, int minArea, int maxArea);
//// inclusive min, exclusive max
//bool contourAreaFilter2(const std::vector<std::vector<cv::Point> >& contours, const std::vector<cv::Vec4i>& hierarchy, int idx, int minArea, int maxArea);
//
//// inclusive min, exclusive max.
//template <typename T>
//cv::Mat bwareaopen(const cv::Mat& binaryImage, int minSize, int maxSize, int connectivity);
//
//template <typename T>
//cv::Mat imhmin(const cv::Mat& image, T h, int connectivity);
//
//cv::Mat_<int> watershed2(const cv::Mat& origImage, const cv::Mat_<float>& image, int connectivity);
//
//template <typename T>
//cv::Mat_<uchar> localMaxima(const cv::Mat& image, int connectivity);
//template <typename T>
//cv::Mat_<uchar> localMinima(const cv::Mat& image, int connectivity);


}


}


#endif /* MORPHOLOGICOPERATION_H_ */
