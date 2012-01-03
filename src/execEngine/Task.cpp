/*
 * Task.cpp
 *
 *  Created on: Aug. 17, 2011
 *      Author: george
 */

#include "Task.h"
#include <stdio.h>

Task::Task()
{
	for(int i = 0; i < ExecEngineConstants::NUM_PROC_TYPES; i++){
		speedups[i] = 1.0;
	}
}


Task::~Task()
{
}

void *Task::getGPUTempData(int tid){
	void * returnDataPtr=NULL;
	if(curExecEngine != NULL){
		returnDataPtr = curExecEngine->getGPUTempData(tid);
	}
	return returnDataPtr;
}

void Task::setSpeedup(int procType, float speedup)
{
	speedups[procType-1] = speedup;

}

float Task::getSpeedup(int procType)
{
	return speedups[procType-1];
}

bool Task::run(int procType, int tid)
{
//	printf("EI Task::run ");
//	if(procType == ExecEngineConstants::CPU){
//		sleep((int)this->getSpeedup(ExecEngineConstants::GPU));
//	}else{
//		sleep((int)this->getSpeedup(ExecEngineConstants::CPU));
//	}
}



