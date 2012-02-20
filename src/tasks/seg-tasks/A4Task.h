/*
 * A4Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef A4TASK_H_
#define A4TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class A4Task: public Task {
public:
	A4Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~A4Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task* next;
	std::string outfilename;

};

}

#endif /* A4TASK_H_ */
