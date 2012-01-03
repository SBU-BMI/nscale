/*
 * B5Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B5Task.h"
#include "HistologicalEntities.h"
#include "B6Task.h"
#include "MorphologicOperations.h"

namespace nscale {

B5Task::B5Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;
}


B5Task::~B5Task() {
	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B5Task::run(int procType) {
	// begin work
	int result;	

	printf("B5\n");


	if (procType == ExecEngineConstants::GPU) {  // GPU
#if defined (HAVE_CUDA)

::cv::gpu::Stream stream;
		::cv::gpu::GpuMat g_input = ::cv::gpu::createContinuous(input.size(), input.type());

		stream.enqueueUpload(input, g_input);
		stream.waitForCompletion();

        ::cv::gpu::GpuMat g_output = ::nscale::gpu::imfillHoles<unsigned char>(g_input, true, 8, stream);
        stream.waitForCompletion();
	
		result = ::nscale::HistologicalEntities::CONTINUE;
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_input.release();
		g_output.release();
#else
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
		CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support");
#endif
	} else if (procType == ExecEngineConstants::CPU) { // CPU
		output = ::nscale::imfillHoles<unsigned char>(input, true, 8);
		result = ::nscale::HistologicalEntities::CONTINUE;
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new B6Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);
	}	
      return true;

}


}
