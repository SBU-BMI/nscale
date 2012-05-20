/*
 * B3Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B3Task.h"
#include "HistologicalEntities.h"
#include "B4Task.h"

namespace nscale {

B3Task::B3Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 4);

}


B3Task::~B3Task() {
//	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B3Task::run(int procType, int tid) {
	// begin work
	int result;	

	printf("B3\n");

#if !defined (WITH_CUDA)
	procType = ExecEngineconstants::CPU;
#endif

	if (procType == ExecEngineConstants::GPU) {  // GPU

::cv::gpu::Stream stream;
		::cv::gpu::GpuMat g_input = ::cv::gpu::createContinuous(input.size(), input.type());

		stream.enqueueUpload(input, g_input);
		stream.waitForCompletion();

                // erode by 1
        // a 3x3 mat with a cross
        unsigned char disk3raw[9] = {
                        0, 1, 0,
                        1, 1, 1,
                        0, 1, 0};
        std::vector<unsigned char> disk3vec(disk3raw, disk3raw+9);
        ::cv::Mat disk3(disk3vec);

        ::cv::gpu::GpuMat g_output = ::nscale::gpu::morphErode<unsigned char>(g_input, disk3, stream);
	
		result = ::nscale::HistologicalEntities::CONTINUE;
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_input.release();
		g_output.release();

	} else if (procType == ExecEngineConstants::CPU) { // CPU
        ::cv::Mat disk3 = getStructuringElement(::cv::MORPH_ELLIPSE, ::cv::Size(3,3));

	        ::cv::Mat twm;
        copyMakeBorder(input, twm, 1, 1, 1, 1, ::cv::BORDER_CONSTANT, ::cv::Scalar(std::numeric_limits<uchar>::max()));
        ::cv::Mat t_nonoverlap = ::cv::Mat::zeros(twm.size(), twm.type());
        erode(twm, t_nonoverlap, disk3);
        output = t_nonoverlap(::cv::Rect(1,1,input.cols, input.rows));
	t_nonoverlap.release();
	twm.release();


		result = ::nscale::HistologicalEntities::CONTINUE;
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new B4Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		if (insertTask(next) != 0) {
			printf("unable to insert task\n");
			return false;
		}
	}	
      return true;

}


}
