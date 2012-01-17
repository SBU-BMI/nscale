/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "cv.h"
#include <fstream>
#include <iostream>
#include <sys/time.h>

namespace cciutils {

const int DEVICE_CPU = 0;
const int DEVICE_MCORE = 1;
const int DEVICE_GPU = 2;

//convert double to unsigned char
inline unsigned char double2uchar(double d){
	double truncate = std::min( std::max(d,(double)0.0), (double)255.0);
	double pt;
	double c = modf(truncate, &pt)>=.5?ceil(truncate):floor(truncate);
	return (unsigned char)c;
}

inline uint64_t ClockGetTime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
 //   timespec ts;
//    clock_gettime(CLOCK_REALTIME, &ts);
	return (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL;
//    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
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



}




#endif /* UTILS_H_ */
