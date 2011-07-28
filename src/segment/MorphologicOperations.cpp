/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include "MorphologicOperations.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <limits>
#include "utils.h"
#include "highgui.h"




namespace nscale {

using namespace cv;


template <typename T>
inline void propagate(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, const T& pval) {
	T qval = output.ptr<T>(y)[x];
	T ival = image.ptr<T>(y)[x];
	if ((qval < pval) && (ival != qval)) {
		output.ptr<T>(y)[x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const uchar&);
template
inline void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const float&);


/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
template <typename T>
Mat imreconstruct(const Mat& seeds, const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);


	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	bool shouldAdd;
	T* oPtr;
	T* oPtrMinus;
	T* oPtrPlus;
	T* iPtr;
	T* iPtrPlus;

	uint64_t t1 = cciutils::ClockGetTime();

	// raster scan
	for (int y = 1; y < maxy; ++y) {

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		for (int x = 1; x < maxx; ++x) {
			xminus = x-1;
			xplus = x+1;
			pval = oPtr[x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			pval = max(pval, max(oPtr[xminus], oPtrMinus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrMinus[xplus], oPtrMinus[xminus]));
			}
			oPtr[x] = min(pval, iPtr[x]);
		}
	}

	// anti-raster scan
	for (int y = maxy-1; y > 0; --y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);

		for (int x = maxx-1; x > 0; --x) {
			xminus = x-1;
			xplus = x+1;

			pval = oPtr[x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = max(pval, max(oPtr[xplus], oPtrPlus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrPlus[xplus], oPtrPlus[xminus]));
			}

			oPtr[x] = min(pval, iPtr[x]);

			// capture the seeds
			shouldAdd = false;
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];
			if ((oPtr[xplus] < pval) && (oPtr[xplus] < iPtr[xplus])) shouldAdd = true;
			if ((oPtrPlus[x] < pval) && (oPtrPlus[x] < iPtrPlus[x])) shouldAdd = true;

			if (connectivity == 8) {
				if ((oPtrPlus[xplus] < pval) && (oPtrPlus[xplus] < iPtrPlus[xplus])) shouldAdd = true;
				if ((oPtrPlus[xminus] < pval) && (oPtrPlus[xminus] < iPtrPlus[xminus])) shouldAdd = true;
			}
			if (shouldAdd) {
				xQ.push(x);
				yQ.push(y);
			}
		}
	}

	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    scan time = " << t2-t1 << "ms" << std::endl;

	// now process the queue.
	T qval, ival;
	int x, y;
	while (!(xQ.empty())) {
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		yminus = y-1;
		yplus = y+1;
		xplus = x+1;

		pval = output.ptr<T>(y)[x];

		// look at the 4 connected components
		if (y > 0) {
			propagate<T>(input, output, xQ, yQ, x, yminus, pval);
		}
		if (y < maxy) {
			propagate<T>(input, output, xQ, yQ, x, yplus, pval);
		}
		if (x > 0) {
			propagate<T>(input, output, xQ, yQ, xminus, y, pval);
		}
		if (x < maxx) {
			propagate<T>(input, output, xQ, yQ, xplus, y, pval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yminus, pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yminus, pval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yplus, pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yplus, pval);
				}

			}
		}
	}


	uint64_t t3 = cciutils::ClockGetTime();
	std::cout << "    queue time = " << t3-t2 << "ms" << std::endl;


	return output(Range(1, maxy), Range(1, maxx));

}





template <typename T>
inline void propagateBinary(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, const T& foreground) {
	if ((output.ptr<T>(y)[x] == 0) && (image.ptr<T>(y)[x] != 0)) {
		output.ptr<T>(y)[x] = foreground;
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const uchar&);
template
inline void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const float&);

/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
Mat imreconstructBinary(const Mat& seeds, const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	T pval, ival;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	bool added;
	T* oPtr;
	T* oPtrPlus;
	T* oPtrMinus;
	T* iPtr;

	uint64_t t1 = cciutils::ClockGetTime();


	// contour pixel determination.  if any neighbor of a 1 pixel is 0, and the image is 1, then boundary
	for (int y = 1; y < maxy; ++y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		for (int x = 1; x < maxx; ++x) {

			pval = oPtr[x];
			ival = iPtr[x];
			added = false;

			if (pval != 0 && ival != 0) {
				xminus = x == 0 ? 0 : x - 1;
				xplus = x == maxx ? maxx : x + 1;

				// 4 connected
				if ((oPtrMinus[x] == 0) ||
						(oPtrPlus[x] == 0) ||
						(oPtr[xplus] == 0) ||
						(oPtr[xminus] == 0)) {
					xQ.push(x);
					yQ.push(y);
					continue;
				}

				// 8 connected

				if (connectivity == 8) {
					if ((oPtrMinus[xminus] == 0) ||
						(oPtrMinus[xplus] == 0) ||
						(oPtrPlus[xminus] == 0) ||
						(oPtrPlus[xplus] == 0)) {
								xQ.push(x);
								yQ.push(y);
								continue;
					}
				}
			}
		}
	}

	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    scan time = " << t2-t1 << "ms" << std::endl;


	// now process the queue.
	T qval;
	T outval = std::numeric_limits<T>::max();
	int x, y;
	while (!(xQ.empty())) {
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		yminus = y-1;
		yplus = y+1;
		xplus = x+1;

		// look at the 4 connected components
		if (y > 0) {
			propagateBinary<T>(input, output, xQ, yQ, x, yminus, outval);
		}
		if (y < maxy) {
			propagateBinary<T>(input, output, xQ, yQ, x, yplus, outval);
		}
		if (x > 0) {
			propagateBinary<T>(input, output, xQ, yQ, xminus, y, outval);
		}
		if (x < maxx) {
			propagateBinary<T>(input, output, xQ, yQ, xplus, y, outval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yminus, outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yminus, outval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yplus, outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yplus, outval);
				}

			}
		}

	}

	uint64_t t3 = cciutils::ClockGetTime();
	std::cout << "    queue time = " << t3-t2 << "ms" << std::endl;

	return output(Range(1, maxy), Range(1, maxx));

}



// Operates on BINARY IMAGES ONLY
template <typename T>
Mat imfill(const Mat& image, const Mat& seeds, bool binary, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

	/* MatLAB imfill code:
	 *     mask = imcomplement(I);
    marker = mask;
    marker(:) = 0;
    marker(locations) = mask(locations);
    marker = imreconstruct(marker, mask, conn);
    I2 = I | marker;
	 */

	Mat mask = std::numeric_limits<T>::max() - image;  // validated

	Mat marker = Mat::zeros(mask.size(), mask.type());

	mask.copyTo(marker, seeds);

	if (binary) marker = imreconstructBinary<T>(marker, mask, connectivity);
	else marker = imreconstruct<T>(marker, mask, connectivity);

	return image | marker;
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat imfillHoles(const Mat& image, bool binary, int connectivity) {
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

	T mn = std::numeric_limits<T>::min();
	T mx = std::numeric_limits<T>::max();
	Rect roi = Rect(1, 1, image.cols, image.rows);

	// copy the input and pad with -inf.
	Mat mask(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, mask, 1, 1, 1, 1, BORDER_CONSTANT, mn);
	// create marker with inf inside and -inf at border, and take its complement
	Mat marker(mask.size(), mask.type());
	Mat marker2(marker, roi);
	marker2 = Scalar(mn);
	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, BORDER_CONSTANT, mx);

	// now do the work...
	mask = mx - mask;
	Mat output;
	if (binary) output = imreconstructBinary<T>(marker, mask, connectivity);
	else output = imreconstruct<T>(marker, mask, connectivity);
	output = mx - output;

	return output(roi);
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat bwselect(const Mat& binaryImage, const Mat& seeds, int connectivity) {
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

	Mat marker = Mat::zeros(seeds.size(), seeds.type());
	binaryImage.copyTo(marker, seeds);

	marker = imreconstructBinary<T>(marker, binaryImage, connectivity);

	return marker & binaryImage;
}

// Operates on BINARY IMAGES ONLY
// ideally, output should be 64 bit unsigned.
//	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
//  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
//  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
Mat_<int> bwlabel(const Mat& binaryImage, int connectivity) {
	CV_Assert(binaryImage.channels() == 1);
	// only works for binary images.

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat_<int> output = Mat_<int>::zeros(binaryImage.size());
	Mat input = binaryImage.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int color = 1;
	// iterate over all top level contours (all siblings, draw with own label color
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
		// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
		drawContours( output, contours, idx, Scalar(color), CV_FILLED, 8, hierarchy );
	}
	return output;
}

// Operates on BINARY IMAGES ONLY
// ideally, output should be 64 bit unsigned.
//	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
//  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
//  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
template <typename T>
Mat bwlabelFiltered(const Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int), int connectivity) {
	// only works for binary images.
	if (contourFilter == NULL) {
		return bwlabel(binaryImage, connectivity);
	}
	CV_Assert(binaryImage.channels() == 1);


	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat output = Mat::zeros(binaryImage.size(), (binaryOutput ? binaryImage.type() : CV_32S));
	Mat input = binaryImage.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (binaryOutput) {
		Scalar color(std::numeric_limits<T>::max());
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
			if (contourFilter(contours, hierarchy, idx)) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				drawContours( output, contours, idx, color, CV_FILLED, 8, hierarchy );
			}
		}

	} else {
		int color = 1;
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
			if (contourFilter(contours, hierarchy, idx)) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				drawContours( output, contours, idx, Scalar(color), CV_FILLED, 8, hierarchy );
			}
		}
	}
	return output;
}

// inclusive min, exclusive max
bool contourAreaFilter(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {

	int area = contourArea(Mat(contours[idx]));
	if (area < minArea) return false;

	int i = hierarchy[idx][2];
	for ( ; i >= 0; i = hierarchy[i][0]) {
		area -= contourArea(Mat(contours[i]));
		if (area < minArea) return false;
	}

	if (area >= maxArea) return false;
	return true;
}

// inclusive min, exclusive max
template <typename T>
Mat bwareaopen(const Mat& binaryImage, int minSize, int maxSize, int connectivity) {
	// only works for binary images.
	CV_Assert(binaryImage.channels() == 1);

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat output = Mat::zeros(binaryImage.size(), binaryImage.type());
	Mat input = binaryImage.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Scalar color(std::numeric_limits<T>::max());
	// iterate over all top level contours (all siblings, draw with own label color
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
		if (contourAreaFilter(contours, hierarchy, idx, minSize, maxSize)) {
			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
			drawContours(output, contours, idx, color, CV_FILLED, 8, hierarchy );
		}
	}
	return output;
}

template <typename T>
Mat imhmin(const Mat& image, T h, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	//	IMHMIN(I,H) suppresses all minima in I whose depth is less than h
	// MatLAB implementation:
	/**
	 *
		I = imcomplement(I);
		I2 = imreconstruct(imsubtract(I,h), I, conn);
		I2 = imcomplement(I2);
	 *
	 */
	T mx = std::numeric_limits<T>::max();
	Mat mask = mx - image;
	Mat marker = mask - h;
	Mat output = imreconstruct<T>(marker, mask, connectivity);
	return mx - output;
}

Mat watershed2(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	/*
	 * MatLAB implementation:
		cc = bwconncomp(imregionalmin(A, conn), conn);
		L = watershed_meyer(A,conn,cc);

	 */
	Mat minima = localMinima<float>(image, connectivity);
	Mat_<int> labels = bwlabel(minima, connectivity);
	Mat image3(image.size(), CV_8UC3);
	cvtColor(image, image3, CV_GRAY2BGR);
	watershed(image3, labels);

	return labels;
}

// only works with integer images
template <typename T>
Mat localMaxima(const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);

	// use morphologic reconstruction.
	Mat marker = image - 1;
	Mat candidates =
			marker < imreconstruct<T>(marker, image, connectivity);
//		marker >= imreconstruct<T>(marker, image, connectivity);
//	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
	//return candidates;

	// now check the candidates
	// first pad the border
	T mn = std::numeric_limits<T>::min();
	T mx = std::numeric_limits<T>::max();
	Mat output(candidates.size() + Size(2,2), candidates.type());
	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);

	int maxy = input.rows-1;
	int maxx = input.cols-1;
	int xminus, xplus;
	T val;
	T *iPtr, *iPtrMinus, *iPtrPlus, *oPtr;
	Rect reg(1, 1, image.cols, image.rows);
	Scalar zero(0);
	Scalar smx(mx);
	Range xrange(1, maxx);
	Range yrange(1, maxy);
	Mat inputBlock = input(yrange, xrange);

	// next iterate over image, and set candidates that are non-max to 0 (via floodfill)
	for (int y = 1; y < maxy; ++y) {

		iPtr = input.ptr<T>(y);
		iPtrMinus = input.ptr<T>(y-1);
		iPtrPlus = input.ptr<T>(y+1);
		oPtr = output.ptr<T>(y);

		for (int x = 1; x < maxx; ++x) {

			// not a candidate, continue.
			if (oPtr[x] > 0) continue;

			xminus = x-1;
			xplus = x+1;

			val = iPtr[x];
			// compare values

			// 4 connected
			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
				std::cout << ".";
				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
				floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
				continue;
			}

			// 8 connected
			if (connectivity == 8) {
				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
					std::cout << "*";
					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
					floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
					continue;
				}
			}

		}
	}
	std::cout << std::endl;
	return output(yrange, xrange) == 0;

}

template <typename T>
Mat localMinima(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	Mat cimage = std::numeric_limits<T>::max() - image;
	return localMaxima<T>(cimage, connectivity);
}


// only works with integer images
template <typename T>
Mat localMaxima2(const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);

	bool flat = true;
	T firstval;
	//	using floodfill
	T mn = std::numeric_limits<T>::min();
	
	// next check for flat image
	MatConstIterator_<T> it = image.begin<T>();
	MatConstIterator_<T> it_end = image.end<T>();
	if (it != it_end) {
		firstval = *it;
		++it;
	} else {
		return Mat::zeros(image.size(), image.type());
	}
	for ( ; it != it_end; ++it) {
		if (*it != firstval) flat = false;
	}
	if (flat) {
		return Mat::ones(image.size(), image.type());  // return 1s when it's flat (instead of type max)
	}
	
	// first pad the border
	Mat output(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, output, 1, 1, 1, 1, BORDER_CONSTANT, mn);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);

	int maxy = input.rows-1;
	int maxx = input.cols-1;
	int xminus, xplus;
	T val;
	T *iPtr, *iPtrMinus, *iPtrPlus, *oPtr;
	Rect reg(1, 1, image.cols, image.rows);
	Scalar zero(0);
	Scalar smn(mn);
	Range xrange(1, maxx);
	Range yrange(1, maxy);

	
	// next iterate over image, and set non-max to MIN (via floodfill)
	for (int y = 1; y < maxy; ++y) {

		iPtr = input.ptr<T>(y);
		iPtrMinus = input.ptr<T>(y-1);
		iPtrPlus = input.ptr<T>(y+1);
		oPtr = output.ptr<T>(y);
		
		for (int x = 1; x < maxx; ++x) {
			xminus = x-1;
			xplus = x+1;
			
			// if already visited, skip to next.
			if (oPtr[x] == mn) continue;
			
			val = iPtr[x];
			// compare values and flood fill.
			
			// 4 connected
			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
				floodFill(output, Point(x, y), smn, &reg, zero, zero, FLOODFILL_FIXED_RANGE | connectivity);
				continue;
			}  // note that if flat region, nothing is changed.  if the flat region is max, it's not going to be changed
			// if flat region is not max, at the edge floodfill will kick in.

			// 8 connected
			if (connectivity == 8) {
				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
					floodFill(output, Point(x, y), smn, &reg, zero, zero, FLOODFILL_FIXED_RANGE | connectivity);
					continue;
				}
			}
		}
	}

	return output(yrange, xrange) > mn;
}

template <typename T>
Mat localMinima2(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	Mat cimage = std::numeric_limits<T>::max() - image;
	return localMaxima2<T>(cimage, connectivity);
}



template Mat imreconstruct<uchar>(const Mat& seeds, const Mat& image, int connectivity);
template Mat imreconstruct<float>(const Mat& seeds, const Mat& image, int connectivity);

template Mat imreconstructBinary<uchar>(const Mat& seeds, const Mat& binaryImage, int connectivity);
template Mat imfill<uchar>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
template Mat imfillHoles<uchar>(const Mat& image, bool binary, int connectivity);
template Mat bwselect<uchar>(const Mat& binaryImage, const Mat& seeds, int connectivity);
template Mat bwlabelFiltered<uchar>(const Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
		int connectivity);
template Mat bwareaopen<uchar>(const Mat& binaryImage, int minSize, int maxSize, int connectivity);
template Mat imhmin(const Mat& image, uchar h, int connectivity);
template Mat imhmin(const Mat& image, float h, int connectivity);
template Mat localMaxima<float>(const Mat& image, int connectivity);
template Mat localMinima<float>(const Mat& image, int connectivity);
template Mat localMaxima2<float>(const Mat& image, int connectivity);
template Mat localMinima2<float>(const Mat& image, int connectivity);
template Mat localMaxima<uchar>(const Mat& image, int connectivity);
template Mat localMinima<uchar>(const Mat& image, int connectivity);
template Mat localMaxima2<uchar>(const Mat& image, int connectivity);
template Mat localMinima2<uchar>(const Mat& image, int connectivity);

}

