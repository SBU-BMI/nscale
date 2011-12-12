/*
 * Task.h
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#ifndef TASK_H_
#define TASK_H_

#include "ExecEngineConstants.h"
#include <unistd.h>

using namespace std;

class Task {
	float speedups[ExecEngineConstants::NUM_PROC_TYPES];

public:
	Task();
	virtual ~Task();
	void setSpeedup(int procType=ExecEngineConstants::GPU, float speedup=1.0);
	float getSpeedup(int procType=ExecEngineConstants::GPU);

	virtual bool run(int procType=ExecEngineConstants::GPU);

};

#endif /* TASK_H */
