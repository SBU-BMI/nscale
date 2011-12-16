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

void Task::setSpeedup(int procType, float speedup)
{
	speedups[procType-1] = speedup;

}

float Task::getSpeedup(int procType)
{
	return speedups[procType-1];
}

bool Task::run(int procType)
{
	printf("EI");
//	if(procType == Constant::CPU){
//		sleep((int)this->getSpeedup(Constant::GPU));
//	}else{
//		sleep((int)this->getSpeedup(Constant::CPU));
//	}
}



