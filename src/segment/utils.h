/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "cv.h"

namespace cciutils {

inline uint64_t ClockGetTime()
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
}

template <typename T>
inline T min()
{
	if (std::numeric_limits<T>::is_integer) {
		return std::numeric_limits<T>::min();
	} else {
		return -std::numeric_limits<T>::max();
	}
}


namespace cv {

using ::cv::Exception;
using ::cv::error;

inline void imwriteRaw(const char *prefix, const ::cv::Mat& img) {
	// write the raw image
	char * filename = new char[128];
	int cols = img.cols;
	int rows = img.rows;
	sprintf(filename, "%s_%d_x_%d.raw", prefix, cols, rows);
	FILE* fid = fopen(filename, "wb");
	const uchar* imgPtr;
	int elSize = img.elemSize();
	for (int j = 0; j < rows; ++j) {
		imgPtr = img.ptr(j);

		fwrite(imgPtr, elSize, cols, fid);
	}
	fclose(fid);

}
template <typename T>
inline ::cv::Mat invert(const ::cv::Mat& img) {
	// write the raw image
	CV_Assert(img.channels() == 1);

	if (std::numeric_limits<T>::is_integer) {

		if (std::numeric_limits<T>::is_signed) {
			::cv::Mat output;
			::cv::bitwise_not(img, output);
			return output + 1;
		} else {
			// unsigned int
			return std::numeric_limits<T>::max() - img;
		}

	} else {
		// floating point type
		return -img;
	}


}

}


}




#endif /* UTILS_H_ */
