/*
 * TaskSleep.h
 *
 *  Created on: Aug 25, 2011
 *      Author: george
 */

#ifndef TASKSLEEP_H_
#define TASKSLEEP_H_

#include "Task.h"

class TaskSleep: public Task {
private:

public:
	TaskSleep();

	virtual ~TaskSleep();

	bool run(int procType=ExecEngineConstants::CPU);
};

#endif /* TASKSLEEP_H_ */
