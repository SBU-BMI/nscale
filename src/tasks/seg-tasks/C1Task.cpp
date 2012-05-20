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
//#include "C2Task.h"
//#include "C3Task.h"
#include <vector>

namespace nscale {

C1Task::C1Task(const ::cv::Mat& image, const ::cv::Mat& in) {
	img = image;
	input = in;
	gray.create(img.size(), CV_8U);
	H.create(img.size(), CV_8U);
	E.create(img.size(), CV_8U);
	next1 = NULL;
	next2 = NULL;
	next3 = NULL;
	next4 = NULL;

	setSpeedup(ExecEngineConstants::GPU, 20);

}


C1Task::~C1Task() {
//	if (next1 != NULL) delete next1;
//	if (next2 != NULL) delete next2;
//	if (next3 != NULL) delete next3;
//	if (next4 != NULL) delete next4;
	gray.release();
	H.release();
	E.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
// just the color deconvolution, then begin invocation of the feature computations.
bool C1Task::run(int procType, int tid) {
	// begin work
	int result;	
	printf("C1\n");

       ::cv::Mat b = (::cv::Mat_<char>(1,3) << 1, 1, 0);
        ::cv::Mat M = (::cv::Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

#if !defined (WITH_CUDA)
	procType = ExecEngineconstants::CPU;
#endif

	if (procType == ExecEngineConstants::GPU) {  // GPU

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

//	//stage the next work
//	if (result == ::nscale::HistologicalEntities::CONTINUE) {
//
//		IplImage iplmask(input);
//		IplImage iplgray(gray);
//
//		// TODO: regionalanalysis needs to be updated to use GPU.
//		RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(&iplmask, &iplgray, true);
//
//
//		std::vector<std::vector<float> > nucleiFeatures;
//
//	// now create the next task
//		next1 = new C2Task(regional, gray, nucleiFeatures);
//	// and invoke it (temporary until hook up the exec engine).
//		if (insertTask(next1) != 0) {
//			printf("unable to insert task\n");
//			return false;
//		}
//	// now create the next task
//		next2 = new C3Task(regional, gray, nucleiFeatures);
//	// and invoke it (temporary until hook up the exec engine).
//		if (insertTask(next2) != 0) {
//			printf("unable to insert task\n");
//			return false;
//		}
//
//	// now create the next task
//		next3 = new C3Task(regional, H, nucleiFeatures);
//	// and invoke it (temporary until hook up the exec engine).
//		if (insertTask(next3) != 0) {
//			printf("unable to insert task\n");
//			return false;
//		}
//
//	// now create the next task
//		next4 = new C3Task(regional, E, nucleiFeatures);
//	// and invoke it (temporary until hook up the exec engine).
//		if (insertTask(next4) != 0) {
//			printf("unable to insert task\n");
//			return false;
//		}
//
//
//
//
//		// clean up
////		delete regional;
//
//	}
      return true;

}


}
