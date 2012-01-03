/*
 * Task.h
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#ifndef TASK_H_
#define TASK_H_

#include <unistd.h>
#include "ExecutionEngine.h"
#include "ExecEngineConstants.h"

using namespace std;

class ExecutionEngine;

class Task {
private:
	float speedups[ExecEngineConstants::NUM_PROC_TYPES];

	friend class ExecutionEngine;
public:

	ExecutionEngine *curExecEngine;
	Task();
	virtual ~Task();
	void setSpeedup(int procType=ExecEngineConstants::GPU, float speedup=1.0);
	float getSpeedup(int procType=ExecEngineConstants::GPU);


	int insertTask(Task *task);
	void *getGPUTempData(int tid);

	// Interface implemented by the end user
	virtual bool run(int procType=ExecEngineConstants::GPU, int tid=0);

};

#endif /* TASK_H */
