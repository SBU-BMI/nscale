/*
 * C2Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef C2TASK_H_
#define C2TASK_H_

#include "Task.h"
#include "RegionalMorphologyAnalysis.h"
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class C2Task: public Task {
public:
	C2Task(RegionalMorphologyAnalysis *reg, const ::cv::Mat& image, std::vector<std::vector<float> >& features);
	virtual ~C2Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	RegionalMorphologyAnalysis *regional;
	std::vector<std::vector<float > > nucleiFeatures;
	Task *next;
};

}

#endif /* C2TASK_H_ */
