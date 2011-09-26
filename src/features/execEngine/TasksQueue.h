/*
 * TasksQueue.h
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#ifndef TASKSQUEUE_H_
#define TASKSQUEUE_H_

#include <map>
#include <list>
#include <semaphore.h>
#include "Task.h"
#include <pthread.h>


class TasksQueue {
protected:
	pthread_mutex_t queueLock;
	sem_t tasksToBeProcessed;

//#ifdef PRIORITY_QUEUE
//	multimap<float, Task*> tasksQueue;
//#else
//	list<Task*> tasksQueue;
//#endif

public:
	TasksQueue();
	virtual ~TasksQueue();

	// These methods are implemented in subclasses according to the type of queue chosen
	virtual bool insertTask(Task* task);
	virtual Task* getTask(int procType=Constant::CPU);

	// Unlock threads that may be waiting at the getTask function
	void releaseThreads(int numThreads);

};

class TasksQueueFCFS: public TasksQueue {
private:
	list<Task*> tasksQueue;

public:
	bool insertTask(Task* task);
	Task* getTask(int procType=Constant::CPU);
};

class TasksQueuePriority: public TasksQueue {
private:
	multimap<float, Task*> tasksQueue;

public:
	bool insertTask(Task* task);
	Task* getTask(int procType=Constant::CPU);

};


#endif /* TASKSQUEUE_H_ */
