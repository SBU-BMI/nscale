/*
 * B6Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef B6TASK_H_
#define B6TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class B6Task: public Task {
public:
	B6Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~B6Task();

	bool run(int procType=ExecEngineConstants::CPU);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task *next;
	std::string outfilename;

};

}

#endif /* B6TASK_H_ */
