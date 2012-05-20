/*
 * B1Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B1Task.h"
#include "HistologicalEntities.h"
#include "B2Task.h"
#include "MorphologicOperations.h"

namespace nscale {

B1Task::B1Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 10);

}


B1Task::~B1Task() {
//	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B1Task::run(int procType, int tid) {
	// begin work
	int result;	

	printf("B1\n");

#if !defined (WITH_CUDA)
	procType = ExecEngineConstants::CPU;
#endif


	if (procType == ExecEngineConstants::GPU) {  // GPU

	::cv::gpu::Stream stream;
	::cv::gpu::GpuMat g_input = ::cv::gpu::createContinuous(input.size(), input.type());

		stream.enqueueUpload(input, g_input);
		stream.waitForCompletion();

        ::cv::gpu::GpuMat g_output = ::nscale::gpu::imfillHoles<unsigned char>(g_input, true, 4, stream);
        stream.waitForCompletion();
	
		result = ::nscale::HistologicalEntities::CONTINUE;
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_input.release();
		g_output.release();

	} else if (procType == ExecEngineConstants::CPU) { // CPU
		output = ::nscale::imfillHoles<unsigned char>(input, true, 4);
		result = ::nscale::HistologicalEntities::CONTINUE;
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new B2Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		if (insertTask(next) != 0) {
			printf("unable to insert task\n");
			return false;
		}
	}	
      return true;

}


}
