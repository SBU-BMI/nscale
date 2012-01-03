/*
 * B4Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "B4Task.h"
#include "HistologicalEntities.h"
#include "B5Task.h"
#include "MorphologicOperations.h"
namespace nscale {

B4Task::B4Task(const ::cv::Mat& image, const ::cv::Mat& in, const std::string& ofn) {
	img = image;
	input = in;
	output.create(img.size(), CV_8U);
	next = NULL;
	outfilename = ofn;

}


B4Task::~B4Task() {
	if (next != NULL) delete next;
	output.release();
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
bool B4Task::run(int procType) {
	// begin work
	int result = ::nscale::HistologicalEntities::CONTINUE;

	printf("B4\n");


       	output = ::nscale::bwareaopen<uchar>(input, 21, 1000, 4);
 
        if (countNonZero(output) == 0) {
                result = ::nscale::HistologicalEntities::NO_CANDIDATES_LEFT;
        }

	


	//stage the next work
	if (result == ::nscale::HistologicalEntities::CONTINUE) {

	// now create the next task
		next = new B5Task(img, output, outfilename);
	// and invoke it (temporary until hook up the exec engine).
		next->run(procType);
	}	
      return true;

}


}
