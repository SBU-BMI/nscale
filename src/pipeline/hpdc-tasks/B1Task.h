/*
 * B1Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef B1TASK_H_
#define B1TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class B1Task: public Task {
public:
	B1Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~B1Task();

	bool run(int procType=ExecEngineConstants::CPU);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task *next;
	std::string outfilename;
};

}

#endif /* B1TASK_H_ */
