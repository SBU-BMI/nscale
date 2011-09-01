/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "cv.h"
#include <fstream>;

namespace cciutils {

const int DEVICE_CPU = 0;
const int DEVICE_MCORE = 1;
const int DEVICE_GPU = 2;


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

template <typename T>
inline bool sameSign(T a, T b) {
	return ((a^b) >= 0);
}

class SimpleCSVLogger {
public :

	void startLogger(const char* name) {};
	void stopLogger() {};
	
	template <typename T>
	void log(const char* eventName, T eventVal) {};
protected :
	ofstream header;
	ofstream value;
};

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


}


}




#endif /* UTILS_H_ */
