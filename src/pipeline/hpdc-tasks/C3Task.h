/*
 * C3Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef C3TASK_H_
#define C3TASK_H_

#include "Task.h"
#include "RegionalMorphologyAnalysis.h"
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class C3Task: public Task {
public:
	C3Task(RegionalMorphologyAnalysis *reg, const ::cv::Mat& image, std::vector<std::vector<float> >& features);
	virtual ~C3Task();

	bool run(int procType=ExecEngineConstants::CPU);

private:

	::cv::Mat img;
	RegionalMorphologyAnalysis *regional;
	std::vector<std::vector<float > > nucleiFeatures;
	Task *next;
};

}

#endif /* C3TASK_H_ */
