/*
 * B2Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B2Task.h"
#include "HistologicalEntities.h"
#include "A4Task.h"
#include "MorphologicOperations.h"

namespace nscale {

B2Task::B2Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 4);

}


B2Task::~B2Task() {
//	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B2Task::run(int procType, int tid) {
	// begin work
	int result;	

	printf("B2\n");

#if !defined (WITH_CUDA)
	procType = ExecEngineConstants::CPU;
#endif

	if (procType == ExecEngineConstants::GPU) {  // GPU

::cv::gpu::Stream stream;
		::cv::gpu::GpuMat g_input = ::cv::gpu::createContinuous(input.size(), input.type());

		stream.enqueueUpload(input, g_input);
		stream.waitForCompletion();


        // a 3x3 mat with a cross
        unsigned char disk3raw[9] = {
                        0, 1, 0,
                        1, 1, 1,
                        0, 1, 0};
        std::vector<unsigned char> disk3vec(disk3raw, disk3raw+9);
        ::cv::Mat disk3(disk3vec);
        // can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
        // because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
        //      morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, ::cv::Point(1,1)); //, ::cv::Point(-1, -1), 1, ::cv::BORDER_REFLECT);
        disk3 = disk3.reshape(1, 3);
//      imwrite("test/out-rcopen-strel.pbm", disk19);
        // filter doesnot check borders.  so need to create border.
        ::cv::gpu::GpuMat g_output = ::nscale::gpu::morphOpen<unsigned char>(g_input, disk3, stream);

		result = ::nscale::HistologicalEntities::CONTINUE;
		stream.enqueueDownload(g_output, output);
		stream.waitForCompletion();

		g_input.release();
		g_output.release();

	} else if (procType == ExecEngineConstants::CPU) { // CPU
        ::cv::Mat disk3 = getStructuringElement(::cv::MORPH_ELLIPSE, ::cv::Size(3,3));

        // can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
        // because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
        //      morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, ::cv::Point(1,1)); //, ::cv::Point(-1, -1), 1, ::cv::BORDER_REFLECT);
        output = ::nscale::morphOpen<unsigned char>(input, disk3);

		result = ::nscale::HistologicalEntities::CONTINUE;
	} else { // error
		printf("ERROR: invalid proc type");
		result = ::nscale::HistologicalEntities::RUNTIME_FAILED;
	}	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new A4Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		if (insertTask(next) != 0) {
			printf("unable to insert task\n");
			return false;
		}
	}	
      return true;

}


}
