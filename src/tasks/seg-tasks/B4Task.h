/*
 * B4Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef B4TASK_H_
#define B4TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class B4Task: public Task {
public:
	B4Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~B4Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task *next;
	std::string outfilename;

};

}

#endif /* B4TASK_H_ */
