/*
 * C1Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef C1TASK_H_
#define C1TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class C1Task: public Task {
public:
	C1Task(const ::cv::Mat& image, const ::cv::Mat& input);
	virtual ~C1Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat gray, H, E;
	Task *next1, *next2, *next3, *next4;
};

}

#endif /* C1TASK_H_ */
