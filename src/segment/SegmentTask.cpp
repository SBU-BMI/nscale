/*
 * SegmentTask.cpp
 *
 *  Created on: Sep 26, 2011
 *      Author: tcpan
 */

#include "SegmentTask.h"
#include "HistologicalEntities.h"

namespace nscale {
using namespace cv;

SegmentTask::SegmentTask(const Mat& img, Mat& out, int mcode, cciutils::SimpleCSVLogger *log, int stg) {
	image = img;
	output = out;
	stage = stg;
	modecode = mcode;
	logger = log;
	fileIO = false;
}

SegmentTask::SegmentTask(const std::string& in, std::string& out, int mcode, cciutils::SimpleCSVLogger *log, int stg) {
	infile = in;
	outfile = out;
	stage = stg;
	modecode = mcode;
	logger = log;
	fileIO = true;
}


SegmentTask::~SegmentTask() {
}


bool SegmentTask::run(int procType) {
	int status = 0;
	switch (modecode) {
	case cciutils::DEVICE_CPU :
	case cciutils::DEVICE_MCORE :
	//				logger.consoleOn();
		if (logger) logger->log("type", "cpu");

	//						std::cout << " segmenting cpu chunk size: " << chunk.size().width << "x" << chunk.size().height << std::endl;
		if (fileIO) {
			status = nscale::HistologicalEntities::segmentNuclei(infile, outfile, logger, stage);
		} else {
			status = nscale::HistologicalEntities::segmentNuclei(image, output, logger, stage);
		}
	//				logger.consoleOff();
		break;
	case cciutils::DEVICE_GPU :
		if (logger) logger->log("type", "gpu");

	//						std::cout << " segmenting cpu chunk size: " << chunk.size().width << "x" << chunk.size().height << std::endl;
		if (fileIO) {
			status = nscale::gpu::HistologicalEntities::segmentNuclei(infile, outfile, logger, stage);
		} else {
			status = nscale::gpu::HistologicalEntities::segmentNuclei(image, output, logger, stage);
		}
		break;
	default :
		break;
	}

	if (status == 0) return true;
	else return false;
}

}
