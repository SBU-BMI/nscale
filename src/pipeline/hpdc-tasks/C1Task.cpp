/*
 * C1Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "C1Task.h"
#include "HistologicalEntities.h"
#include "PixelOperations.h"
#include "RegionalMorphologyAnalysis.h"
#include "C2Task.h"
#include "C3Task.h"
#include <vector>

namespace nscale {

C1Task::C1Task(const ::cv::Mat& image, const ::cv::Mat& in) {
	img = image;
	input = in;
	gray.create(img.size(), CV_8U);
	H.create(img.size(), CV_8U);
	E.create(img.size(), CV_8U);
	next = NULL;
}


C1Task::~C1Task() {
	if (next != NULL) delete next;
	gray.release();
	H.release();
	E.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
// just the color deconvolution, then begin invocation of the feature computations.
bool C1Task::run(int procType) {
	// begin work
	int result;	
	printf("C1\n");

       ::cv::Mat b = (::cv::Mat_<char>(1,3) << 1, 1, 0);
        ::cv::Mat M = (::cv::Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);


	if (procType == ExecEngineConstants::GPU) {  // GPU
#if defined (HAVE_CUDA)

		::cv::gpu::Stream stream;
		::cv::gpu::GpuMat g_img = ::cv::gpu::createContinuous(img.size(), img.type());
		::cv::gpu::GpuMat g_H = ::cv::gpu::createContinuous(H.size(), H.type()); 
		::cv::gpu::GpuMat g_E = ::cv::gpu::createContinuous(E.size(), E.type()); 
		stream.enqueueUpload(img, g_img);
		stream.waitForCompletion();

        ::cv::gpu::GpuMat g_gray = ::nscale::gpu::PixelOperations::bgr2gray(g_img, stream);
	::nscale::gpu::PixelOperations::ColorDeconv(g_img, M, b, g_H, g_E, stream);
        stream.waitForCompletion();
	
		result = ::nscale::HistologicalEntities::CONTINUE;
		stream.enqueueDownload(g_gray, gray);
		stream.enqueueDownload(g_H, H);
		stream.enqueueDownload(g_E, E);
		stream.waitForCompletion();

		g_img.release();
		g_gray.release();
		g_H.release();
		g_E.release();
#else
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
		CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support");
#endif

	} else if (procType == ExecEngineConstants::CPU) { // CPU

		gray = ::nscale::PixelOperations::bgr2gray(img);
		::nscale::PixelOperations::ColorDeconv(img, M, b, H, E);

		result = ::nscale::HistologicalEntities::CONTINUE;
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	

	M.release();
	b.release();

	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

		IplImage iplmask(input);
		IplImage iplgray(gray);

		// TODO: regionalanalysis needs to be updated to use GPU.
		RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(&iplmask, &iplgray, true);


		std::vector<std::vector<float> > nucleiFeatures;

	// now create the next task
		next = new C2Task(regional, gray, nucleiFeatures);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);

	// now create the next task
		next = new C3Task(regional, gray, nucleiFeatures);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);


	// now create the next task
		next = new C3Task(regional, H, nucleiFeatures);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);


	// now create the next task
		next = new C3Task(regional, E, nucleiFeatures);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);


		


		// clean up
		delete regional;

	}	
      return true;

}


}
