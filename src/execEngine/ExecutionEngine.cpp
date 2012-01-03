/*
 * ExecutionEngine.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#include "ExecutionEngine.h"


ExecutionEngine::ExecutionEngine(int cpuThreads, int gpuThreads, int queueType, int gpuTempDataSize) {
	if(queueType ==ExecEngineConstants::FCFS_QUEUE){
		tasksQueue = new TasksQueueFCFS(cpuThreads, gpuThreads);
	}else{
		tasksQueue = new TasksQueuePriority(cpuThreads, gpuThreads);
	}
	threadPool = new ThreadPool(tasksQueue);
	threadPool->createThreadPool(cpuThreads, NULL, gpuThreads, NULL, gpuTempDataSize);
}

ExecutionEngine::~ExecutionEngine() {
	delete threadPool;
	delete tasksQueue;
}

void *ExecutionEngine::getGPUTempData(int tid){
	return threadPool->getGPUTempData(tid);
}

bool ExecutionEngine::insertTask(Task *task)
{
	task->curExecEngine = this;
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

void ExecutionEngine::waitUntilMinQueuedTask(int numberQueuedTasks)
{
	if(numberQueuedTasks < 0) numberQueuedTasks = 0;

	// Loop waiting the number of tasks queued decrease
	while(numberQueuedTasks < tasksQueue->getSize()){
		usleep(100000);
	}

}

