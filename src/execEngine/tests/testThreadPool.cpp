/*
 * testQueue.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */


#include "TasksQueue.h"
#include "ThreadPool.h"

int main(){
	TasksQueue tasksQueue;
	ThreadPool *tp = new ThreadPool(&tasksQueue);

	for(int i = 0; i < 20; i++){
		Task *auxTask = new Task();
		auxTask->setSpeedup(ExecEngineConstants::GPU, 1.0);
		tasksQueue.insertTask(auxTask);
	}

	for(int i = 0; i < 20; i++){
		Task *auxTask = new Task();
		auxTask->setSpeedup(ExecEngineConstants::GPU, 6.0);
		tasksQueue.insertTask(auxTask);
	}

	tp->createThreadPool(1, NULL, 1);

	delete tp;

	return 0;
}
