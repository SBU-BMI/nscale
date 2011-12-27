/*
 * TaskFeature.h
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#ifndef TASKFEATURE_H_
#define TASKFEATURE_H_

#include "Task.h"
#include "RegionalMorphologyAnalysis.h"

class TaskFeature: public Task {
private:
	RegionalMorphologyAnalysis *regional;

public:
	TaskFeature(string mask, string img);
	TaskFeature(IplImage *mask, IplImage *img);

	virtual ~TaskFeature();

	bool run(int procType=Constant::GPU);
};

#endif /* TASKFEATURE_H_ */
