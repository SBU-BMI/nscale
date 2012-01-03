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
}

S1Task::~S1Task() {
	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool S1Task::run(int procType) {
	// begin work
	::cv::Mat temp(img.size(), CV_8U);  // temporary until the code could be fixed.
	int result;	

	printf("S1\n");

	if (procType == ExecEngineConstants::GPU) {// GPU
#if defined (HAVE_GPU)

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
#else
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
		CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support");
#endif

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
		next = new B1Task(img, output, outfilename);


		next->run(procType);
	}

	
	return true;
}


}
