/*
 * B3Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef B3TASK_H_
#define B3TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class B3Task: public Task {
public:
	B3Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~B3Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task *next;
	std::string outfilename;

};

}

#endif /* B3TASK_H_ */
