/*
 * C3Task.cpp
 * generate the first pass nuclei candidates for filtering in next step.
 *  NOTE:  rely on memory for passing data.  so is memory bound.
 *
 *  Created on: Dec 29, 2011
 *      Author: tcpan
 */

#include "C3Task.h"
#include "HistologicalEntities.h"
#include "RegionalMorphologyAnalysis.h"

namespace nscale {

C3Task::C3Task(RegionalMorphologyAnalysis *reg, const ::cv::Mat& image, std::vector<std::vector<float> >& features) {
	img = image;
	regional = reg;
	nucleiFeatures = features;
	next = NULL;

	setSpeedup(ExecEngineConstants::GPU, 5);

}


C3Task::~C3Task() {
//	if (next != NULL) delete next;
}

// does not keep data in GPU memory yet.  no appropriate flag to show that data is on GPU, so that execEngine can try to reuse.
// just the color deconvolution, then begin invocation of the feature computations.
bool C3Task::run(int procType, int tid) {
	// begin work
	IplImage image(img);

	printf("C3\n");


	regional->doCytoplasmPipelineFeatures(nucleiFeatures, &image);
      return true;

}


}
