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


using namespace cv;


namespace nscale {

struct PixelLocation {
	int x;
	int y;
};


template <typename T>
inline void propagate(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, const T& pval) {
	T qval = output.ptr(y)[x];
	T ival = image.ptr(y)[x];
	if ((qval < pval) && (ival != qval)) {
		output.ptr(y)[x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const uchar&);


/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
template <typename T>
Mat imreconstruct(const Mat& seeds, const Mat& image, int conn) {
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
	for (int y = 1; y < maxy; y++) {

		oPtr = output.ptr(y);
		oPtrMinus = output.ptr(y-1);
		iPtr = input.ptr(y);

		for (int x = 1; x < maxx; x++) {
			xminus = x-1;
			xplus = x+1;
			pval = oPtr[x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			pval = max(pval, max(oPtr[xminus], oPtrMinus[x]));

			if (conn == 8) {
				pval = max(pval, max(oPtrMinus[xplus], oPtrMinus[xminus]));
			}
			oPtr[x] = min(pval, iPtr[x]);
		}
	}

	// anti-raster scan
	for (int y = maxy-1; y > 0; y--) {
		oPtr = output.ptr(y);
		oPtrPlus = output.ptr(y+1);
		oPtrMinus = output.ptr(y-1);
		iPtr = input.ptr(y);
		iPtrPlus = input.ptr(y+1);

		for (int x = maxx-1; x > 0; x--) {
			xminus = x-1;
			xplus = x+1;

			pval = oPtr[x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = max(pval, max(oPtr[xplus], oPtrPlus[x]));

			if (conn == 8) {
				pval = max(pval, max(oPtrPlus[xplus], oPtrPlus[xminus]));
			}

			oPtr[x] = min(pval, iPtr[x]);

			// capture the seeds
			shouldAdd = false;
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];
			if ((oPtr[xplus] < pval) && (oPtr[xplus] < iPtr[xplus])) shouldAdd = true;
			if ((oPtrPlus[x] < pval) && (oPtrPlus[x] < iPtrPlus[x])) shouldAdd = true;

			if (conn == 8) {
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

		pval = output.ptr(y)[x];

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
		if (conn == 8) {

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
	if ((output.ptr(y)[x] == 0) && (image.ptr(y)[x] != 0)) {
		output.ptr(y)[x] = foreground;
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, const uchar&);


/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
Mat imreconstructBinary(const Mat& seeds, const Mat& image, int conn) {
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
	for (int y = 1; y < maxy; y++) {
		oPtr = output.ptr(y);
		oPtrPlus = output.ptr(y+1);
		oPtrMinus = output.ptr(y-1);
		iPtr = input.ptr(y);

		for (int x = 1; x < maxx; x++) {

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

				if (conn == 8) {
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
	PixelLocation p;
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
		if (conn == 8) {

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
Mat imfillBinary(Mat binaryImage, Mat seeds, int connectivity=8) {

	/* MatLAB imfill code:
	 *     mask = imcomplement(I);
    marker = mask;
    marker(:) = 0;
    marker(locations) = mask(locations);
    marker = imreconstruct(marker, mask, conn);
    I2 = I | marker;
	 */

	T mx = std::numeric_limits<T>::max();
	Mat mask = mx - binaryImage;  // validated

	Mat marker = Mat::zeros(mask.size(), mask.type());

	mask.copyTo(marker, seeds);

	marker = imreconstructBinary<uchar>(marker, mask, connectivity);

	return binaryImage | marker;
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat imfillHolesBinary(Mat binaryImage, int connectivity=8) {

	/* MatLAB imfill code:
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
	Rect roi = Rect(1, 1, binaryImage.cols, binaryImage.rows);

	// copy the input and pad with -inf.
	Mat mask(binaryImage.size() + Size(2,2), binaryImage.type());
	copyMakeBorder(binaryImage, mask, 1, 1, 1, 1, BORDER_CONSTANT, mn);
	// create marker with inf inside and -inf at border, and take its complement
	Mat marker(mask.size(), mask.type());
	Mat marker2(marker, roi);
	marker2 = Scalar(mn);
	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, BORDER_CONSTANT, mx);

	// now do the work...
	mask = mx - mask;
	Mat output = imreconstructBinary<T>(marker, mask, connectivity);
	output = mx - output;

	return output(roi);
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat imfillHoles(Mat image, int connectivity=8) {

	/* MatLAB imfill code:
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
	Mat output = imreconstruct<T>(marker, mask, connectivity);
	output = mx - output;

	return output(roi);
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat bwselectBinary(Mat binaryImage, Mat seeds, int connectivity=8) {
	// only works for binary images.  ~I and MAX-I are the same....

	/** adopted from bwselect and imfill
	 * bwselet:
	 * seed_indices = sub2ind(size(BW), r(:), c(:));
		BW2 = imfill(~BW, seed_indices, n);
		BW2 = BW2 & BW;
	 *
	 * imfill:
	 * see below.
	 */

	Mat marker = Mat::zeros(seeds.size(), seeds.type());
	binaryImage.copyTo(marker, seeds);

	marker = imreconstructBinary<uchar>(marker, binaryImage, connectivity);

	return marker & binaryImage;
}


template Mat bwselectBinary<uchar>(Mat image, Mat seeds, int connectivity);


template Mat imreconstruct<uchar>(const Mat&, const Mat&, int);
template Mat imreconstructBinary<uchar>(const Mat&, const Mat&, int);
template Mat imfillBinary<uchar>(Mat image, Mat seeds, int connectivity);
template Mat imfillHoles<uchar>(Mat image, int connectivity);
template Mat imfillHolesBinary<uchar>(Mat image, int connectivity);


}

