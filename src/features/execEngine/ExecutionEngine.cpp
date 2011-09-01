/*
 * ExecutionEngine.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#include "ExecutionEngine.h"


ExecutionEngine::ExecutionEngine(int cpuThreads, int gpuThreads, int queueType) {
	if(queueType == FCFS_QUEUE){
		tasksQueue = new TasksQueueFCFS();
	}else{
		tasksQueue = new TasksQueuePriority();
	}
	threadPool = new ThreadPool(tasksQueue);
	threadPool->createThreadPool(cpuThreads, NULL, gpuThreads);
}

ExecutionEngine::~ExecutionEngine() {
	delete threadPool;
	delete tasksQueue;
}

bool ExecutionEngine::insertTask(Task *task)
{
	return tasksQueue->insertTask(task);
}


Task *ExecutionEngine::getTask(int procType)
{
	return tasksQueue->getTask(procType);
}

void ExecutionEngine::startupExecution()
{
	threadPool->initExecution();
}

void ExecutionEngine::endExecution()
{

	tasksQueue->releaseThreads(threadPool->getGPUThreads() + threadPool->getCPUThreads());
	delete threadPool;
	threadPool = NULL;
}













