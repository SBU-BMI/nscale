/*
 * TaskFeature.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#include "TaskSleep.h"

TaskSleep::TaskSleep() {
}

TaskSleep::~TaskSleep() {
}

bool TaskSleep::run(int procType)
{
	if(procType == 1){
		sleep(5);
	}else{

		sleep(1);
	}
	return true;
}



