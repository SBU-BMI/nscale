/*
 * S1Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "S1Task.h"
#include "HistologicalEntities.h"
#include "B1Task.h"

namespace nscale {

S1Task::S1Task(const ::cv::Mat& image, const std::string& ofn) {
	img = image;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 2);

}

S1Task::~S1Task() {
//	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool S1Task::run(int procType, int tid) {
	// begin work
	::cv::Mat temp(img.size(), CV_8U);  // temporary until the code could be fixed.
	int result;	

	printf("S1\n");

#if !defined (WITH_CUDA)
	procType = ExecEngineConstants::CPU;
#endif


	if (procType == ExecEngineConstants::GPU) {// GPU

		::cv::gpu::Stream stream;
		::cv::gpu::GpuMat g_img = ::cv::gpu::createContinuous(img.size(), img.type());
		::cv::gpu::GpuMat g_output = ::cv::gpu::createContinuous(img.size(), CV_8U);
		stream.enqueueUpload(img, g_img);
		stream.waitForCompletion();

		result = ::nscale::gpu::HistologicalEntities::plFindNucleusCandidates(g_img, g_output, stream, temp, NULL, -1);
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_img.release();
		g_output.release();

	} else if (procType == ExecEngineConstants::CPU) { // CPU
		result = ::nscale::HistologicalEntities::plFindNucleusCandidates(img, output, temp, NULL, -1);
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	

	temp.release();

	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {
	// now create the next task
	// and invoke it (temporary until hook up the exec engine).

		printf("inserting B1\n");
		next = new B1Task(img, output, outfilename);
		if (next == NULL) {
			printf("ERROR:  did not create B1 correctly\n");
		}

//		int x = insertTask(next);
//		printf("insert task: %d\n", x);
//		if (x != 0) {
//			return false;
//		}

		this->curExecEngine->insertTask(next);
		printf("inserted B1\n");
	}

	
	return true;
}


}
