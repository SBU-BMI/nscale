/*
 * S1Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef S1TASK_H_
#define S1TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class S1Task: public Task {
public:
	S1Task(const ::cv::Mat& image, const std::string& ofn);
	virtual ~S1Task();

	bool run(int procType=ExecEngineConstants::CPU);

private:

	::cv::Mat img;
	::cv::Mat output;
	Task* next;
	std::string outfilename;

};

}

#endif /* S1TASK_H_ */
