/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include <algorithm>
#include <queue>
#include <iostream>
#include <limits>
#include "highgui.h"

#include "utils.h"
#include "MorphologicOperations.h"
#include "PixelOperations.h"


namespace nscale {

using namespace cv;


template <typename T>
inline void propagate(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, T* iPtr, T* oPtr, const T& pval) {
	T qval = oPtr[x];
	T ival = iPtr[x];
	if ((qval < pval) && (ival != qval)) {
		oPtr[x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, uchar* iPtr, uchar* oPtr, const uchar&);
template
inline void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, float* iPtr, float* oPtr, const float&);


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

	T pval, preval;
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
	T* iPtrMinus;

	uint64_t t1 = cciutils::ClockGetTime();

	// raster scan
	for (int y = 1; y < maxy; ++y) {

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		preval = oPtr[0];
		for (int x = 1; x < maxx; ++x) {
			xminus = x-1;
			xplus = x+1;
			pval = oPtr[x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			pval = max(pval, max(preval, oPtrMinus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrMinus[xplus], oPtrMinus[xminus]));
			}
			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;
		}
	}

	// anti-raster scan
	int count = 0;
	for (int y = maxy-1; y > 0; --y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);

		preval = oPtr[maxx];
		for (int x = maxx-1; x > 0; --x) {
			xminus = x-1;
			xplus = x+1;

			pval = oPtr[x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = max(pval, max(preval, oPtrPlus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrPlus[xplus], oPtrPlus[xminus]));
			}

			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;

			// capture the seeds
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];

			if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||
					(oPtrPlus[x] < min(pval, iPtrPlus[x]))) {
				xQ.push(x);
				yQ.push(y);
				++count;
				continue;
			}

			if (connectivity == 8) {
				if ((oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) ||
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus]))) {
					xQ.push(x);
					yQ.push(y);
					++count;
					continue;
				}
			}
		}
	}

	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queue entries."<< std::endl;

	// now process the queue.
	T qval, ival;
	int x, y;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;

		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(yplus);
		oPtrMinus = output.ptr<T>(yminus);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(yplus);
		iPtrMinus = input.ptr<T>(yminus);

		pval = oPtr[x];

		// look at the 4 connected components
		if (y > 0) {
			propagate<T>(input, output, xQ, yQ, x, yminus, iPtrMinus, oPtrMinus, pval);
		}
		if (y < maxy) {
			propagate<T>(input, output, xQ, yQ, x, yplus, iPtrPlus, oPtrPlus,pval);
		}
		if (x > 0) {
			propagate<T>(input, output, xQ, yQ, xminus, y, iPtr, oPtr,pval);
		}
		if (x < maxx) {
			propagate<T>(input, output, xQ, yQ, xplus, y, iPtr, oPtr,pval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yminus, iPtrMinus, oPtrMinus, pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yminus, iPtrMinus, oPtrMinus, pval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yplus, iPtrPlus, oPtrPlus,pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yplus, iPtrPlus, oPtrPlus,pval);
				}

			}
		}
	}


	uint64_t t3 = cciutils::ClockGetTime();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;


	return output(Range(1, maxy), Range(1, maxx));

}

inline void propagateUchar(int *irev, int *ifwd,
		int& x, int offset, uchar* iPtr, uchar* oPtr, uchar& pval) {
    uchar val1 = oPtr[x];
    uchar ival = iPtr[x];
    int val2 = pval < ival ? pval : ival;
    if (val1 < val2) {
      if (val1 != 0) {  // if the neighbor's value is going to be replaced, remove the neighbor from list
        ifwd[irev[offset]] = ifwd[offset];
        if (ifwd[offset] >= 0)
          irev[ifwd[offset]] = irev[offset];
      }
      oPtr[x] = val2;  // replace the value
      irev[offset] = -val2;  // and insert into the list...
      ifwd[offset] = irev[-val2];
      irev[-val2] = offset;
      if (ifwd[offset] >= 0)
        irev[ifwd[offset]] = offset;
    }
}

Mat imreconstructUChar(const Mat& seeds, const Mat& image, int connectivity) {

	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8U);

	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	int width = input.cols;
	int height = input.rows;

	  int ix,iy,ox,oy,offset, currentP;
	  int currentQ;  // current value in downhill
	  int pixPerImg=width*height;
	  uchar val, maxVal = 0;
	  int val1;
	  int *istart,*irev,*ifwd;

	  double mmin, mmax;
	  minMaxLoc(seeds, &mmin, &mmax);
	  maxVal = (int)mmax;

	  // create the downhill list
	  istart = (int*)malloc((maxVal+pixPerImg*2)*sizeof(int));
	  irev = istart+maxVal;
	  ifwd = irev+pixPerImg;
	  // initialize the heads of the lists
	  for (offset = -maxVal; offset < 0; offset++)
	    irev[offset] = offset;

	  // populate the lists with pixel locations - essentially sorting the image by pixel values
	  // backward traversal here will result in forward traversal in the next step.
	  MatIterator_<uchar> mend = output.end<uchar>();
	  for (offset = pixPerImg-1, --mend; offset >= 0; --mend, --offset) {
		  val = *mend;
		  if (val > 0) {
			  val1 = -val;
			  irev[offset] = val1;  // set the end of the list
			  ifwd[offset] = irev[val1];  // move the head of the list into ifwd
			  irev[val1] = offset;   // insert the new head
			  if (ifwd[offset] >= 0)  // if the list was not previously empty
				  irev[ifwd[offset]] = offset;  // then also set the irev for previous head to the current head
		  }
	  }

	  // now do the processing
	  int xminus, xplus, yminus, yplus;
	  int maxx = width - 1;
	  int maxy = height - 1;
	  uchar pval;
	  int x, y;
	  uchar *oPtr, *oPtrPlus, *oPtrMinus, *iPtr, *iPtrPlus, *iPtrMinus;
	  for (currentQ = -maxVal; currentQ < 0; ++currentQ) {
	    currentP = irev[currentQ];   // get the head of the list for the curr value
	    while (currentP >= 0) {  // non empty list
	      irev[currentQ] = ifwd[currentP];  // pop the "stack"
	      irev[currentP] = currentQ;   // remove the end.
	      x = currentP%width;  // get the current position
	      y = currentP/width;
	      //std::cout << "x, y = " << x << ", " << y << std::endl;

			xminus = x-1;
			xplus = x+1;
			yminus = y-1;
			yplus = y+1;

			oPtr = output.ptr<uchar>(y);
			oPtrPlus = output.ptr<uchar>(yplus);
			oPtrMinus = output.ptr<uchar>(yminus);
			iPtr = input.ptr<uchar>(y);
			iPtrPlus = input.ptr<uchar>(yplus);
			iPtrMinus = input.ptr<uchar>(yminus);

			pval = oPtr[x];

			// look at the 4 connected components
			if (y > 0) {
				propagateUchar(irev, ifwd, x, x+yminus*width, iPtrMinus, oPtrMinus, pval);
			}
			if (y < maxy) {
				propagateUchar(irev, ifwd, x, x+yplus*width, iPtrPlus, oPtrPlus, pval);
			}
			if (x > 0) {
				propagateUchar(irev, ifwd, xminus, xminus+y*width, iPtr, oPtr, pval);
			}
			if (x < maxx) {
				propagateUchar(irev, ifwd, xplus, xplus+y*width, iPtr, oPtr, pval);
			}

			// now 8 connected
			if (connectivity == 8) {

				if (y > 0) {
					if (x > 0) {
						propagateUchar(irev, ifwd, xminus, xminus+yminus*width, iPtrMinus, oPtrMinus, pval);
					}
					if (x < maxx) {
						propagateUchar(irev, ifwd, xplus, xplus+yminus*width, iPtrMinus, oPtrMinus, pval);
					}

				}
				if (y < maxy) {
					if (x > 0) {
						propagateUchar(irev, ifwd, xminus, xminus+yplus*width, iPtrPlus, oPtrPlus, pval);
					}
					if (x < maxx) {
						propagateUchar(irev, ifwd, xplus, xplus+yplus*width, iPtrPlus, oPtrPlus, pval);
					}

				}
			}


	      currentP = irev[currentQ];
	    }
	  }
	  free(istart);

	return output(Range(1, maxy), Range(1, maxx));

}



template <typename T>
inline void propagateBinary(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, T* iPtr, T* oPtr, const T& foreground) {
	if ((oPtr[x] == 0) && (iPtr[x] != 0)) {
		oPtr[x] = foreground;
		xQ.push(x);
		yQ.push(y);
	}
}

template
inline void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, uchar* iPtr, uchar* oPtr, const uchar&);
template
inline void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, float* iPtr, float* oPtr, const float&);

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
	T* oPtr;
	T* oPtrPlus;
	T* oPtrMinus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

	uint64_t t1 = cciutils::ClockGetTime();

	int count = 0;
	// contour pixel determination.  if any neighbor of a 1 pixel is 0, and the image is 1, then boundary
	for (int y = 1; y < maxy; ++y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		for (int x = 1; x < maxx; ++x) {

			pval = oPtr[x];
			ival = iPtr[x];

			if (pval != 0 && ival != 0) {
				xminus = x - 1;
				xplus = x + 1;

				// 4 connected
				if ((oPtrMinus[x] == 0) ||
						(oPtrPlus[x] == 0) ||
						(oPtr[xplus] == 0) ||
						(oPtr[xminus] == 0)) {
					xQ.push(x);
					yQ.push(y);
//					++count;
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
//								++count;
								continue;
					}
				}
			}
		}
	}

	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queued "<< std::endl;


	// now process the queue.
	T qval;
	T outval = std::numeric_limits<T>::max();
	int x, y;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		yminus = y-1;
		yplus = y+1;
		xplus = x+1;

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		oPtrPlus = output.ptr<T>(y+1);
		iPtr = input.ptr<T>(y);
		iPtrMinus = input.ptr<T>(y-1);
		iPtrPlus = input.ptr<T>(y+1);

		// look at the 4 connected components
		if (y > 0) {
			propagateBinary<T>(input, output, xQ, yQ, x, yminus, iPtrMinus, oPtrMinus, outval);
		}
		if (y < maxy) {
			propagateBinary<T>(input, output, xQ, yQ, x, yplus, iPtrPlus, oPtrPlus, outval);
		}
		if (x > 0) {
			propagateBinary<T>(input, output, xQ, yQ, xminus, y, iPtr, oPtr, outval);
		}
		if (x < maxx) {
			propagateBinary<T>(input, output, xQ, yQ, xplus, y, iPtr, oPtr, outval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yminus, iPtrMinus, oPtrMinus, outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yminus, iPtrMinus, oPtrMinus, outval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yplus, iPtrPlus, oPtrPlus,outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yplus, iPtrPlus, oPtrPlus,outval);
				}

			}
		}

	}

	uint64_t t3 = cciutils::ClockGetTime();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queued" << std::endl;

	return output(Range(1, maxy), Range(1, maxx));

}



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

	Mat mask = nscale::PixelOperations::invert<T>(image);  // validated

	Mat marker = Mat::zeros(mask.size(), mask.type());

	mask.copyTo(marker, seeds);

	if (binary) marker = imreconstructBinary<T>(marker, mask, connectivity);
	else marker = imreconstruct<T>(marker, mask, connectivity);

	return image | marker;
}

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

	T mn = cciutils::min<T>();
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
	mask = nscale::PixelOperations::invert<T>(mask);

	uint64_t t1 = cciutils::ClockGetTime();
	Mat output;
	if (binary) {
		imwrite("test/in-fillholes-bin-marker.pgm", marker);
		imwrite("test/in-fillholes-bin-mask.pgm", mask);
		output = imreconstructBinary<T>(marker, mask, connectivity);
	} else {
		imwrite("test/in-fillholes-gray-marker.pgm", marker);
		imwrite("test/in-fillholes-gray-mask.pgm", mask);
		output = imreconstruct<T>(marker, mask, connectivity);
	}
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    imfill hole imrecon took " << t2-t1 << "ms" << std::endl;

	output = nscale::PixelOperations::invert<T>(output);

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
Mat_<int> bwlabel(const Mat& binaryImage, bool contourOnly, int connectivity) {
	CV_Assert(binaryImage.channels() == 1);
	// only works for binary images.

	int lineThickness = CV_FILLED;
	if (contourOnly) lineThickness = 1;

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

//	Mat_<int> output = Mat_<int>::zeros(binaryImage.size());
//	Mat input = binaryImage.clone();
	Mat_<int> output = Mat_<int>::zeros(binaryImage.size() + Size(2,2));
	Mat input(output.size(), binaryImage.type());
	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	std::cout << "num contours = " << contours.size() << std::endl;

	int color = 1;
	uint64_t t1 = cciutils::ClockGetTime();
	// iterate over all top level contours (all siblings, draw with own label color
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
		// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
		drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
	}
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "    bwlabel drawing took " << t2-t1 << "ms" << std::endl;

	return output(Rect(1,1,binaryImage.cols, binaryImage.rows));
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
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
		bool contourOnly, int connectivity) {
	// only works for binary images.
	if (contourFilter == NULL) {
		return bwlabel(binaryImage, contourOnly, connectivity);
	}
	CV_Assert(binaryImage.channels() == 1);

	int lineThickness = CV_FILLED;
	if (contourOnly) lineThickness = 1;

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat output = Mat::zeros(binaryImage.size(), (binaryOutput ? binaryImage.type() : CV_32S));
	Mat input = binaryImage.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	if (binaryOutput) {
		Scalar color(std::numeric_limits<T>::max());
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
			if (contourFilter(contours, hierarchy, idx)) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				drawContours( output, contours, idx, color, lineThickness, connectivity, hierarchy );
			}
		}

	} else {
		int color = 1;
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
			if (contourFilter(contours, hierarchy, idx)) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
			}
		}
	}
	return output;
}

// inclusive min, exclusive max
bool contourAreaFilter(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {

	int area = contourArea(contours[idx]);
	int circum = contours[idx].size() / 2 + 1;

	area += circum;

	if (area < minArea) return false;

	int i = hierarchy[idx][2];
	for ( ; i >= 0; i = hierarchy[i][0]) {
		area -= (contourArea(contours[i]) + contours[i].size() / 2 + 1);
		if (area < minArea) return false;
	}

	if (area >= maxArea) return false;
	//std::cout << idx << " total area = " << area << std::endl;

	return true;
}


/*
// inclusive min, exclusive max
// TODO:  still not right...
bool contourAreaFilter2(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {

	uint64_t area = ScanlineOperations::getContourArea(contours, hierarchy, idx);

	//std::cout << idx << " total area = " << area << std::endl;

	if (area < minArea || area >= maxArea) return false;
	else return true;

}
*/


// inclusive min, exclusive max
template <typename T>
Mat bwareaopen(const Mat& binaryImage, int minSize, int maxSize, int connectivity) {
	// only works for binary images.
	CV_Assert(binaryImage.channels() == 1);
	CV_Assert(minSize > 0);
	CV_Assert(maxSize > 0);

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat_<T> output = Mat_<T>::zeros(binaryImage.size());
	Mat input = binaryImage.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	std::cout << "num contours = " << contours.size() << std::endl;

	Scalar color(std::numeric_limits<T>::max());
	// iterate over all top level contours (all siblings, draw with own label color
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
		if (contourAreaFilter(contours, hierarchy, idx, minSize, maxSize)) {
			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
			drawContours(output, contours, idx, color, CV_FILLED, connectivity, hierarchy );
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
	Mat mask = nscale::PixelOperations::invert<T>(image);
	Mat marker = mask - h;
	Mat output = imreconstruct<T>(marker, mask, connectivity);
	return nscale::PixelOperations::invert<T>(output);
}

// input should have foreground > 0, and 0 for background
Mat_<int> watershed2(const Mat& origImage, const Mat_<float>& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	/*
	 * MatLAB implementation:
		cc = bwconncomp(imregionalmin(A, conn), conn);
		L = watershed_meyer(A,conn,cc);

	 */

	Mat minima = localMinima<float>(image, connectivity);
	Mat_<int> labels = bwlabel(minima, true, connectivity);

	// convert to grayscale
	// now scale, shift, clear background.
	double mmin, mmax;
	minMaxLoc(image, &mmin, &mmax);
	std::cout << " image: min=" << mmin << " max=" << mmax<< std::endl;
	double range = (mmax - mmin);
	double scaling = std::numeric_limits<uchar>::max() / range;
	Mat shifted(image.size(), CV_8U);
	image.convertTo(shifted, CV_8U, scaling, 0.0);


	Mat image3(shifted.size(), CV_8UC3);
	cvtColor(shifted, image3, CV_GRAY2BGR);

	watershed(origImage, labels);

	mmin, mmax;
	minMaxLoc(labels, &mmin, &mmax);
	std::cout << " watershed: min=" << mmin << " max=" << mmax<< std::endl;


	return labels;
}

// only works with integer images
template <typename T>
Mat_<uchar> localMaxima(const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);

	// use morphologic reconstruction.
	Mat marker = image - 1;
	Mat_<uchar> candidates =
			marker < imreconstruct<T>(marker, image, connectivity);
//	candidates marked as 0 because floodfill with mask will fill only 0's
//	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
	//return candidates;

	// now check the candidates
	// first pad the border
	T mn = cciutils::min<T>();
	T mx = std::numeric_limits<uchar>::max();
	Mat_<uchar> output(candidates.size() + Size(2,2));
	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);

	int maxy = input.rows-1;
	int maxx = input.cols-1;
	int xminus, xplus;
	T val;
	T *iPtr, *iPtrMinus, *iPtrPlus;
	uchar *oPtr;
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
		oPtr = output.ptr<uchar>(y);

		for (int x = 1; x < maxx; ++x) {

			// not a candidate, continue.
			if (oPtr[x] > 0) continue;

			xminus = x-1;
			xplus = x+1;

			val = iPtr[x];
			// compare values

			// 4 connected
			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
				floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
				continue;
			}

			// 8 connected
			if (connectivity == 8) {
				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
					floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
					continue;
				}
			}

		}
	}
	return output(yrange, xrange) == 0;  // similar to bitwise not.

}

template <typename T>
Mat_<uchar> localMinima(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	Mat cimage = nscale::PixelOperations::invert<T>(image);
	return localMaxima<T>(cimage, connectivity);
}




template Mat imreconstruct<uchar>(const Mat& seeds, const Mat& image, int connectivity);
template Mat imreconstruct<float>(const Mat& seeds, const Mat& image, int connectivity);

template Mat imreconstructBinary<uchar>(const Mat& seeds, const Mat& binaryImage, int connectivity);
template Mat imfill<uchar>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
template Mat imfillHoles<uchar>(const Mat& image, bool binary, int connectivity);
template Mat bwselect<uchar>(const Mat& binaryImage, const Mat& seeds, int connectivity);
template Mat bwlabelFiltered<uchar>(const Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
		bool contourOnly, int connectivity);
template Mat bwareaopen<uchar>(const Mat& binaryImage, int minSize, int maxSize, int connectivity);
template Mat imhmin(const Mat& image, uchar h, int connectivity);
template Mat imhmin(const Mat& image, float h, int connectivity);
template Mat_<uchar> localMaxima<float>(const Mat& image, int connectivity);
template Mat_<uchar> localMinima<float>(const Mat& image, int connectivity);
template Mat_<uchar> localMaxima<uchar>(const Mat& image, int connectivity);
template Mat_<uchar> localMinima<uchar>(const Mat& image, int connectivity);

}

