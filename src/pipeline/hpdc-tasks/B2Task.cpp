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

namespace nscale {

B2Task::B2Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;
}


B2Task::~B2Task() {
	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B2Task::run(int procType) {
	// begin work
	int result;	

	printf("B2\n");


	if (procType == ExecEngineConstants::GPU) {  // GPU
#if defined (HAVE_CUDA)

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
        ::cv::gpu::GpuMat g_t_seg_nohole;
        copyMakeBorder(g_input, g_t_seg_nohole, 1,1,1,1, ::cv::Scalar(std::numeric_limits<unsigned char>::max()), stream);
        ::cv::gpu::GpuMat g_t_seg_erode(g_t_seg_nohole.size(), g_t_seg_nohole.type());
        erode(g_t_seg_nohole, g_t_seg_erode, disk3, ::cv::Point(-1,-1), 1, stream);
        ::cv::gpu::GpuMat g_seg_erode = g_t_seg_erode(::cv::Rect(1, 1, g_input.cols, g_input.rows));
        ::cv::gpu::GpuMat g_t_seg_erode2;
        copyMakeBorder(g_seg_erode, g_t_seg_erode2, 1,1,1,1, ::cv::Scalar(std::numeric_limits<unsigned char>::min()), stream);
        ::cv::gpu::GpuMat g_t_seg_open(g_t_seg_erode2.size(), g_t_seg_erode2.type());
        dilate(g_t_seg_erode2, g_t_seg_open, disk3, ::cv::Point(-1,-1), 1, stream);
        ::cv::gpu::GpuMat g_output = g_t_seg_open(::cv::Rect(1, 1, g_input.cols, g_input.rows));
        stream.waitForCompletion();
        g_t_seg_open.release();
        g_t_seg_erode2.release();
        g_seg_erode.release();
        g_t_seg_erode.release();
        g_t_seg_nohole.release();


	
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
        ::cv::Mat disk3 = getStructuringElement(::cv::MORPH_ELLIPSE, ::cv::Size(3,3));

        // can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
        // because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
        //      morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, ::cv::Point(1,1)); //, ::cv::Point(-1, -1), 1, ::cv::BORDER_REFLECT);
        ::cv::Mat t_seg_nohole;
        copyMakeBorder(input, t_seg_nohole, 1, 1, 1, 1, ::cv::BORDER_CONSTANT, std::numeric_limits<uchar>::max());
        ::cv::Mat t_seg_erode = ::cv::Mat::zeros(t_seg_nohole.size(), t_seg_nohole.type());
        erode(t_seg_nohole, t_seg_erode, disk3);
        ::cv::Mat seg_erode = t_seg_erode(::cv::Rect(1, 1, input.cols, input.rows));
        ::cv::Mat t_seg_erode2;
        copyMakeBorder(seg_erode,t_seg_erode2, 1, 1, 1, 1, ::cv::BORDER_CONSTANT, std::numeric_limits<uchar>::min());
        ::cv::Mat t_seg_open = ::cv::Mat::zeros(t_seg_erode2.size(), t_seg_erode2.type());
        dilate(t_seg_erode2, t_seg_open, disk3);
        output = t_seg_open(::cv::Rect(1,1,input.cols, input.rows));
        t_seg_open.release();
        t_seg_erode2.release();
        seg_erode.release();
        t_seg_erode.release();
        t_seg_nohole.release();
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
		next->run(procType);
	}	
      return true;

}


}
