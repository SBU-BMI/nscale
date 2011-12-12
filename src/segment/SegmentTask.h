/*
 * SegmentTask.h
 *
 *  Created on: Sep 26, 2011
 *      Author: tcpan
 */

#ifndef SEGMENTTASK_H_
#define SEGMENTTASK_H_

#include "execEngine/Task.h"
#include "cv.h"
#include "utils.h"
#include <string.h>

namespace nscale {

using namespace cv;

class SegmentTask: public Task {
protected:
	Mat image;
	Mat output;
	cciutils::SimpleCSVLogger *logger;
	int modecode;
	int stage;
	std::string infile;
	std::string outfile;
	bool fileIO;


public:
	SegmentTask(const Mat& img, Mat& out, int mcode, cciutils::SimpleCSVLogger *log=NULL, int stg=-1);
	SegmentTask(const std::string& in, std::string& out, int mcode, cciutils::SimpleCSVLogger *log=NULL, int stg= -1);
	virtual ~SegmentTask();

	bool run(int procType=ExecEngineConstants::CPU);
};

}

#endif /* SEGMENTTASK_H_ */
