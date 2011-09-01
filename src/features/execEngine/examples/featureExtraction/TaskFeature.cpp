/*
 * TaskFeature.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#include "TaskFeature.h"

TaskFeature::TaskFeature(string mask, string img) {
	regional = new RegionalMorphologyAnalysis(mask.c_str(), img.c_str());
}
TaskFeature::TaskFeature(IplImage *mask, IplImage *img){
	regional = new RegionalMorphologyAnalysis(mask, img);
}

TaskFeature::~TaskFeature() {
	delete regional;
}

bool TaskFeature::run(int procType)
{
	printf("TaskFeature::run\n");
	regional->doCoocPropsBlob(Constant::ANGLE_0, procType);
	regional->doCoocPropsBlob(Constant::ANGLE_45, procType);
	regional->doCoocPropsBlob(Constant::ANGLE_90, procType);
	regional->doCoocPropsBlob(Constant::ANGLE_135, procType);

	regional->doCoocPropsBlob(Constant::ANGLE_0, procType, false);
	regional->doCoocPropsBlob(Constant::ANGLE_45, procType, false);
	regional->doCoocPropsBlob(Constant::ANGLE_90, procType, false);
	regional->doCoocPropsBlob(Constant::ANGLE_135, procType, false);

	if(procType == 2){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_0, procType, false);
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_45, procType, false);
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_90, procType, false);
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_135, procType, false);
}



