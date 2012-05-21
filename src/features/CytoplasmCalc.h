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
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>
#include "Operators.h"

namespace nscale{

class CytoplasmCalc {
public:
	static int* calcCytoplasm(int& cytoDataSize, const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask);

};

namespace gpu{
class CytoplasmCalc {
public:

};

}
}// end nscale
#endif /* CYTOPLASMCALC_H_ */
