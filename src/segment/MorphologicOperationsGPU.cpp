/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */
#define HAVE_CUDA 1


#include <algorithm>
#include <queue>
#include <iostream>
#include <limits>
#include "highgui.h"

#include "utils.h"
#include "MorphologicOperations.h"
#include "PixelOperations.h"

#include "precomp.hpp"

#include "cuda/imreconstruct_int_kernel.cuh"
#include "cuda/imreconstruct_float_kernel.cuh"
#include "cuda/imreconstruct_binary_kernel.cuh"
#include "cuda/reconstruction_kernel.cuh"


namespace nscale {

namespace gpu {

using namespace cv;
using namespace cv::gpu;


/**
 * based on implementation from Pavlo
 */
template <typename T>
GpuMat imreconstruct(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1);

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstruct<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//	return output;

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	if (std::numeric_limits<T>::is_integer) {
	    iter = imreconstructIntCaller<T>(marker.data, mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	} else {
		iter = imreconstructFloatCaller<T>(marker, mask, connectivity, StreamAccessor::getStream(stream));
	}
    stream.waitForCompletion();
    mask.release();
    // get the result out
    return marker;
}
//template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
//template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);



/**
 * based on implementation from Pavlo
 */
template <typename T>
GpuMat imreconstruct2(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1);

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstruct<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//	return output;

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	iter = reconstruction_by_dilation_kernel(marker.data, mask.data, seeds.cols, seeds.rows, 1);
    stream.waitForCompletion();

	mask.release();
    return marker;
}
template GpuMat imreconstruct2<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct2<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);



/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
GpuMat imreconstructBinary(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {

	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_8UC1);

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();

    iter = imreconstructBinaryCaller<T>(marker, mask, connectivity, StreamAccessor::getStream(stream));
    stream.waitForCompletion();
    mask.release();

	return marker;

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstructBinary<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//
//	return output;

}
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);


//
//template <typename T>
//Mat imfill(const Mat& image, const Mat& seeds, bool binary, int connectivity) {
//	CV_Assert(image.channels() == 1);
//	CV_Assert(seeds.channels() == 1);
//
//	/* MatLAB imfill code:
//	 *     mask = imcomplement(I);
//    marker = mask;
//    marker(:) = 0;
//    marker(locations) = mask(locations);
//    marker = imreconstruct(marker, mask, conn);
//    I2 = I | marker;
//	 */
//
//	Mat mask = nscale::gpu::PixelOperations::invert<T>(image, stream);  // validated
//
//	Mat marker = Mat::zeros(mask.size(), mask.type());
//
//	mask.copyTo(marker, seeds);
//
//	if (binary) marker = imreconstructBinary<T>(marker, mask, connectivity);
//	else marker = imreconstruct<T>(marker, mask, connectivity);
//
//	return image | marker;
//}
//
template <typename T>
GpuMat imfillHoles(const GpuMat& image, bool binary, int connectivity, Stream& stream) {
	CV_Assert(image.channels() == 1);

	/* MatLAB imfill hole code:
    if islogical(I)
        mask = uint8(I);
    else
        mask = I;
    end
    mask = padarray(mask, ones(1,ndims(mask)), -Inf, 'both');

    marker = mask;
    idx = cell(1,ndims(I));
    for k = 1:ndims(I)
        idx{k} = 2:(size(marker,k) - 1);
    end
    marker(idx{:}) = Inf;

    mask = imcomplement(mask);
    marker = imcomplement(marker);
    I2 = imreconstruct(marker, mask, conn);
    I2 = imcomplement(I2);
    I2 = I2(idx{:});

    if islogical(I)
        I2 = I2 ~= 0;
    end
	 */

	T mn = cciutils::min<T>();
	T mx = std::numeric_limits<T>::max();
	Rect roi = Rect(1, 1, image.cols, image.rows);

	// copy the input and pad with -inf.
	GpuMat mask2;
	copyMakeBorder(image, mask2, 1, 1, 1, 1, Scalar(mn), stream);
	// create marker with inf inside and -inf at border, and take its complement
	GpuMat marker;
	GpuMat marker2(image.size(), image.type());
	stream.enqueueMemSet(marker2, Scalar(mn));

	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, Scalar(mx), stream);

	// now do the work...
	GpuMat mask = nscale::gpu::PixelOperations::invert<T>(mask2, stream);
	stream.waitForCompletion();
	marker2.release();
	mask2.release();

	uint64_t t1 = cciutils::ClockGetTime();
	GpuMat output2;
//	if (binary) output2 = imreconstructBinary<T>(marker, mask, connectivity, stream);
//	else output2 = imreconstruct<T>(marker, mask, connectivity, stream);
	output2 = imreconstruct2<T>(marker, mask, connectivity, stream);
	stream.waitForCompletion();
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    imfill hole imrecon took " << t2-t1 << "ms" << std::endl;
	stream.waitForCompletion();
	marker.release();
	mask.release();

	GpuMat output3 = nscale::gpu::PixelOperations::invert<T>(output2, stream);
	stream.waitForCompletion();
	output2.release();
	GpuMat output(output3, roi);
	output3.release();

	return output;
}
template GpuMat imfillHoles<unsigned char>(const GpuMat&, bool, int, Stream&);

// Operates on BINARY IMAGES ONLY
template <typename T>
GpuMat bwselect(const GpuMat& binaryImage, const GpuMat& seeds, int connectivity, Stream& stream) {
	CV_Assert(binaryImage.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	// only works for binary images.  ~I and max-I are the same....

	/** adopted from bwselect and imfill
	 * bwselet:
	 * seed_indices = sub2ind(size(BW), r(:), c(:));
		BW2 = imfill(~BW, seed_indices, n);
		BW2 = BW2 & BW;
	 *
	 * imfill:
	 * see imfill function.
	 */

	// general simplified bwselect

	// since binary, seeds already have the same values as binary images
	// at the selected places.  If not, the marker will be forced to 0 by imrecon.

	GpuMat marker = imreconstruct2<T>(seeds, binaryImage, connectivity, stream);
//	GpuMat marker = imreconstructBinary<T>(seeds, binaryImage, connectivity, stream);

	// no need to and between marker and binaryImage - since marker is always <= binary image
	return marker;
}
template GpuMat bwselect<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
//
//// Operates on BINARY IMAGES ONLY
//// ideally, output should be 64 bit unsigned.
////	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
//// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
////  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
//// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
////  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
//Mat_<int> bwlabel(const Mat& binaryImage, bool contourOnly, int connectivity) {
//	CV_Assert(binaryImage.channels() == 1);
//	// only works for binary images.
//
//	int lineThickness = CV_FILLED;
//	if (contourOnly) lineThickness = 1;
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
////	Mat_<int> output = Mat_<int>::zeros(binaryImage.size());
////	Mat input = binaryImage.clone();
//	Mat_<int> output = Mat_<int>::zeros(binaryImage.size() + Size(2,2));
//	Mat input(output.size(), binaryImage.type());
//	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//	std::cout << "num contours = " << contours.size() << std::endl;
//
//	int color = 1;
//	uint64_t t1 = cciutils::ClockGetTime();
//	// iterate over all top level contours (all siblings, draw with own label color
//	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
//		// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//		drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
//	}
//	uint64_t t2 = cciutils::ClockGetTime();
//	std::cout << "    bwlabel drawing took " << t2-t1 << "ms" << std::endl;
//
//	return output(Rect(1,1,binaryImage.cols, binaryImage.rows));
//}
//
//// Operates on BINARY IMAGES ONLY
//// ideally, output should be 64 bit unsigned.
////	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
//// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
////  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
//// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
////  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
//template <typename T>
//Mat bwlabelFiltered(const Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
//		bool contourOnly, int connectivity) {
//	// only works for binary images.
//	if (contourFilter == NULL) {
//		return bwlabel(binaryImage, contourOnly, connectivity);
//	}
//	CV_Assert(binaryImage.channels() == 1);
//
//	int lineThickness = CV_FILLED;
//	if (contourOnly) lineThickness = 1;
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
//	Mat output = Mat::zeros(binaryImage.size(), (binaryOutput ? binaryImage.type() : CV_32S));
//	Mat input = binaryImage.clone();
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//
//	if (binaryOutput) {
//		Scalar color(std::numeric_limits<T>::max());
//		// iterate over all top level contours (all siblings, draw with own label color
//		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
//			if (contourFilter(contours, hierarchy, idx)) {
//				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//				drawContours( output, contours, idx, color, lineThickness, connectivity, hierarchy );
//			}
//		}
//
//	} else {
//		int color = 1;
//		// iterate over all top level contours (all siblings, draw with own label color
//		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
//			if (contourFilter(contours, hierarchy, idx)) {
//				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//				drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
//			}
//		}
//	}
//	return output;
//}
//
//// inclusive min, exclusive max
//bool contourAreaFilter(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {
//
//	int area = contourArea(contours[idx]);
//	int circum = contours[idx].size() / 2 + 1;
//
//	area += circum;
//
//	if (area < minArea) return false;
//
//	int i = hierarchy[idx][2];
//	for ( ; i >= 0; i = hierarchy[i][0]) {
//		area -= (contourArea(contours[i]) + contours[i].size() / 2 + 1);
//		if (area < minArea) return false;
//	}
//
//	if (area >= maxArea) return false;
////	std::cout << idx << " total area = " << area << std::endl;
//
//	return true;
//}
//
//// inclusive min, exclusive max
//template <typename T>
//Mat bwareaopen(const Mat& binaryImage, int minSize, int maxSize, int connectivity) {
//	// only works for binary images.
//	CV_Assert(binaryImage.channels() == 1);
//	CV_Assert(minSize > 0);
//	CV_Assert(maxSize > 0);
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
//	Mat_<T> output = Mat_<T>::zeros(binaryImage.size());
//	Mat input = binaryImage.clone();
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//	std::cout << "num contours = " << contours.size() << std::endl;
//
//	Scalar color(std::numeric_limits<T>::max());
//	// iterate over all top level contours (all siblings, draw with own label color
//	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
//		if (contourAreaFilter(contours, hierarchy, idx, minSize, maxSize)) {
//			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//			drawContours(output, contours, idx, color, CV_FILLED, connectivity, hierarchy );
//		}
//	}
//	return output;
//}
//
//template <typename T>
//Mat imhmin(const Mat& image, T h, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	//	IMHMIN(I,H) suppresses all minima in I whose depth is less than h
//	// MatLAB implementation:
//	/**
//	 *
//		I = imcomplement(I);
//		I2 = imreconstruct(imsubtract(I,h), I, conn);
//		I2 = imcomplement(I2);
//	 *
//	 */
//	Mat mask = nscale::gpu::PixelOperations::invert<T>(image, stream);
//	Mat marker = mask - h;
//	Mat output = imreconstruct<T>(marker, mask, connectivity);
//	return nscale::gpu::PixelOperations::invert<T>(output,stream);
//}
//
//// input should have foreground > 0, and 0 for background
//Mat_<int> watershed2(const Mat& origImage, const Mat_<float>& image, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	/*
//	 * MatLAB implementation:
//		cc = bwconncomp(imregionalmin(A, conn), conn);
//		L = watershed_meyer(A,conn,cc);
//
//	 */
//
//	Mat minima = localMinima<float>(image, connectivity);
//	Mat_<int> labels = bwlabel(minima, true, connectivity);
//
//	// convert to grayscale
//	// now scale, shift, clear background.
//	double mmin, mmax;
//	minMaxLoc(image, &mmin, &mmax);
//	std::cout << " image: min=" << mmin << " max=" << mmax<< std::endl;
//	double range = (mmax - mmin);
//	double scaling = std::numeric_limits<uchar>::max() / range;
//	Mat shifted(image.size(), CV_8U);
//	image.convertTo(shifted, CV_8U, scaling, 0.0);
//
//
//	Mat image3(shifted.size(), CV_8UC3);
//	cvtColor(shifted, image3, CV_GRAY2BGR);
//
//	watershed(origImage, labels);
//
//	mmin, mmax;
//	minMaxLoc(labels, &mmin, &mmax);
//	std::cout << " watershed: min=" << mmin << " max=" << mmax<< std::endl;
//
//
//	return labels;
//}
//
//// only works with integer images
//template <typename T>
//Mat_<uchar> localMaxima(const Mat& image, int connectivity) {
//	CV_Assert(image.channels() == 1);
//
//	// use morphologic reconstruction.
//	Mat marker = image - 1;
//	Mat_<uchar> candidates =
//			marker < imreconstruct<T>(marker, image, connectivity);
////	candidates marked as 0 because floodfill with mask will fill only 0's
////	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
//	//return candidates;
//
//	// now check the candidates
//	// first pad the border
//	T mn = cciutils::min<T>();
//	T mx = std::numeric_limits<uchar>::max();
//	Mat_<uchar> output(candidates.size() + Size(2,2));
//	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
//	Mat input(image.size() + Size(2,2), image.type());
//	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);
//
//	int maxy = input.rows-1;
//	int maxx = input.cols-1;
//	int xminus, xplus;
//	T val;
//	T *iPtr, *iPtrMinus, *iPtrPlus;
//	uchar *oPtr;
//	Rect reg(1, 1, image.cols, image.rows);
//	Scalar zero(0);
//	Scalar smx(mx);
//	Range xrange(1, maxx);
//	Range yrange(1, maxy);
//	Mat inputBlock = input(yrange, xrange);
//
//	// next iterate over image, and set candidates that are non-max to 0 (via floodfill)
//	for (int y = 1; y < maxy; ++y) {
//
//		iPtr = input.ptr<T>(y);
//		iPtrMinus = input.ptr<T>(y-1);
//		iPtrPlus = input.ptr<T>(y+1);
//		oPtr = output.ptr<uchar>(y);
//
//		for (int x = 1; x < maxx; ++x) {
//
//			// not a candidate, continue.
//			if (oPtr[x] > 0) continue;
//
//			xminus = x-1;
//			xplus = x+1;
//
//			val = iPtr[x];
//			// compare values
//
//			// 4 connected
//			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
//				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
//				floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
//				continue;
//			}
//
//			// 8 connected
//			if (connectivity == 8) {
//				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
//					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
//					floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
//					continue;
//				}
//			}
//
//		}
//	}
//	return output(yrange, xrange) == 0;  // similar to bitwise not.
//
//}
//
//template <typename T>
//Mat_<uchar> localMinima(const Mat& image, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	Mat cimage = nscale::gpu::PixelOperations::invert<T>(image, stream);
//	return localMaxima<T>(cimage, connectivity);
//}




//template Mat imfill<uchar>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
//template Mat bwselect<uchar>(const Mat& binaryImage, const Mat& seeds, int connectivity);
//template Mat bwlabelFiltered<uchar>(const Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
//		bool contourOnly, int connectivity);
//template Mat bwareaopen<uchar>(const Mat& binaryImage, int minSize, int maxSize, int connectivity);
//template Mat imhmin(const Mat& image, uchar h, int connectivity);
//template Mat imhmin(const Mat& image, float h, int connectivity);
//template Mat_<uchar> localMaxima<float>(const Mat& image, int connectivity);
//template Mat_<uchar> localMinima<float>(const Mat& image, int connectivity);
//template Mat_<uchar> localMaxima<uchar>(const Mat& image, int connectivity);
//template Mat_<uchar> localMinima<uchar>(const Mat& image, int connectivity);

}

}

