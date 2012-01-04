/*
 * B5Task.h
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#ifndef B5TASK_H_
#define B5TASK_H_

#include "Task.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class B5Task: public Task {
public:
	B5Task(const ::cv::Mat& image, const ::cv::Mat& input, const std::string& ofn);
	virtual ~B5Task();

	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

private:

	::cv::Mat img;
	::cv::Mat input;
	::cv::Mat output;
	Task *next;
	std::string outfilename;

};

}

#endif /* B5TASK_H_ */
