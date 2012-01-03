/*
 * ExecutionEngine.h
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#ifndef EXECUTIONENGINE_H_
#define EXECUTIONENGINE_H_

#include "ThreadPool.h"
#include "TasksQueue.h"

//#define FCFS_QUEUE	1
//#define PRIORITY_QUEUE	2
class Task;
class TasksQueue;
class ThreadPool;

class ExecutionEngine {

private:
	TasksQueue *tasksQueue;
	ThreadPool *threadPool;
	bool state; // waiting to execute; executing; done;

	Task* getTask(int procType=ExecEngineConstants::CPU);

public:
	ExecutionEngine(int cpuThreads, int gpuThreads, int queueType=ExecEngineConstants::FCFS_QUEUE, int gpuTempDataSize=0);
	virtual ~ExecutionEngine();

	bool insertTask(Task* task);

	void *getGPUTempData(int tid);

	// Execution engine will start computation of tasks
	void startupExecution();

	// No more tasks will be queued, and whenever the tasks
	// already queued are computed the exec. engine will finish.
	// This is a blocking call.
	void endExecution();

	void waitUntilMinQueuedTask(int number_queued_tasks);
};

#endif /* EXECUTIONENGINE_H_ */
