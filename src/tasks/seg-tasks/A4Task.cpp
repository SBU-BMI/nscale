/*
 * A4Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "A4Task.h"
#include "HistologicalEntities.h"
#include "B3Task.h"

namespace nscale {

A4Task::A4Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 2);
}

A4Task::~A4Task() {
//	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool A4Task::run(int procType, int tid) {
	// begin work
	int result;	

	printf("A4\n");

#if !defined (WITH_CUDA)
	procType = ExecEngineConstants::CPU;
#endif


	if (procType == ExecEngineConstants::GPU) {  // GPU

		::cv::gpu::Stream stream;

		::cv::gpu::GpuMat g_img = ::cv::gpu::createContinuous(img.size(), img.type());
		::cv::gpu::GpuMat g_input = ::cv::gpu::createContinuous(input.size(), input.type());

		::cv::gpu::GpuMat g_output = ::cv::gpu::createContinuous(input.size(), input.type());
		stream.enqueueUpload(img, g_img);
		stream.enqueueUpload(input, g_input);
		stream.waitForCompletion();

		result = ::nscale::gpu::HistologicalEntities::plSeparateNuclei(g_img, g_input, g_output, stream, NULL, NULL);
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_img.release();
		g_input.release();
		g_output.release();

	} else if (procType == ExecEngineConstants::CPU) { // CPU
		result = ::nscale::HistologicalEntities::plSeparateNuclei(img, input, output, NULL, NULL);
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new B3Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		if (insertTask(next) != 0) {
			printf("unable to insert task\n");
			return false;
		}
	}	
    return true;

}


}
