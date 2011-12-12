/*
 * Task.h
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#ifndef TASK_H_
#define TASK_H_

#include "Constants.h"
#include <unistd.h>

using namespace std;

class Task {
	float speedups[Constant::NUM_PROC_TYPES];

public:
	Task();
	virtual ~Task();
	void setSpeedup(int procType=Constant::GPU, float speedup=1.0);
	float getSpeedup(int procType=Constant::GPU);

	virtual bool run(int procType=Constant::GPU);

};

#endif /* TASK_H */
