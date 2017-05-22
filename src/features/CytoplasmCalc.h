/*
 * CytoplasmCalc.h
 *
 *  Created on: May 10, 2012
 *      Author: gteodor
 */

#ifndef CYTOPLASMCALC_H_
#define CYTOPLASMCALC_H_

// Includes to use opencv2/GPU
#include "opencv2/opencv.hpp"

#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif 

#ifdef _MSC_VER
#include "time_win.h"
#else
#include <sys/time.h>
#endif
#include "Operators.h"

namespace nscale{

class CytoplasmCalc {
public:
	static int* calcCytoplasm(int& cytoDataSize, const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask);

};

#ifdef WITH_CUDA
namespace gpu{
class CytoplasmCalc {
public:

};

}
#endif 

}// end nscale
#endif /* CYTOPLASMCALC_H_ */
