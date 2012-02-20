/*
 * B6Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B6Task.h"
#include "HistologicalEntities.h"
#include "C1Task.h"

namespace nscale {

B6Task::B6Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

	setSpeedup(ExecEngineConstants::GPU, 1);

}


B6Task::~B6Task() {
//	if (next != NULL) delete next;
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B6Task::run(int procType, int tid) {
	// begin work
	printf("B6\n");


	// placeholder for bounding box detection.
	output = input;
	printf("writing to [%s]\n", outfilename.c_str());
	if (!outfilename.empty()) ::cv::imwrite(outfilename, output);


	int result = ::nscale::HistologicalEntities::CONTINUE;

       	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new C1Task(img, output);
	// and invoke it (temporary until hook up the exec engine).
		if (insertTask(next) != 0) {
			printf("unable to insert task\n");
			return false;
		}
	}	
      return true;

}


}
