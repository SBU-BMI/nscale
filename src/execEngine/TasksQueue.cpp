/*
 * TasksQueue.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#include "TasksQueue.h"


TasksQueue::TasksQueue() {
	pthread_mutex_init(&queueLock, NULL);
	sem_init(&tasksToBeProcessed, 0, 0);
}

TasksQueue::~TasksQueue() {
	pthread_mutex_destroy(&queueLock);
	sem_destroy(&tasksToBeProcessed);
}
bool TasksQueue::insertTask(Task *task)
{
	return true;
}

Task *TasksQueue::getTask(int procType)
{
	return NULL;
}

int TasksQueue::getSize()
{
	return 0;
}


bool TasksQueueFCFS::insertTask(Task *task)
{
	pthread_mutex_lock(&queueLock);

	tasksQueue.push_back(task);

	pthread_mutex_unlock(&queueLock);
	sem_post(&tasksToBeProcessed);
	return true;
}

Task *TasksQueueFCFS::getTask(int procType)
{
	Task *retTask = NULL;
	sem_wait(&tasksToBeProcessed);
	pthread_mutex_lock(&queueLock);

	if(tasksQueue.size() > 0){
		retTask = tasksQueue.front();
//		tasksQueue.pop_front();
#ifdef	LOAD_BALANCING
		if(ExecEngineConstants::CPU == procType){
			float taskSpeedup = retTask->getSpeedup(ExecEngineConstants::GPU);
			if( this->gpuThreads* taskSpeedup > tasksQueue.size()){
				retTask = NULL;
			}
		}
#endif
		if(retTask != NULL)
			tasksQueue.pop_front();
	}
	pthread_mutex_unlock(&queueLock);
	return retTask;
}

int TasksQueueFCFS::getSize()
{
	int number_tasks = 0;
	pthread_mutex_lock(&queueLock);

	number_tasks = tasksQueue.size();

	pthread_mutex_unlock(&queueLock);

	return number_tasks;
}


bool TasksQueuePriority::insertTask(Task *task)
{
	pthread_mutex_lock(&queueLock);

	float taskSpeedup = task->getSpeedup(ExecEngineConstants::GPU);
	tasksQueue.insert(pair<float,Task*>(taskSpeedup, task));

	pthread_mutex_unlock(&queueLock);
	sem_post(&tasksToBeProcessed);
	return true;
}
Task *TasksQueuePriority::getTask(int procType)
{
	Task *retTask = NULL;
	sem_wait(&tasksToBeProcessed);
	pthread_mutex_lock(&queueLock);

	int taskQueueSize = tasksQueue.size();

	if(taskQueueSize > 0){
		multimap<float, Task*>::iterator it;

		if(procType == ExecEngineConstants::GPU){
			it = tasksQueue.end();
			it--;
			retTask = (*it).second;
			tasksQueue.erase(it);
		}else{
			it = tasksQueue.begin();
			retTask = (*it).second;
#ifdef	LOAD_BALANCING
			float taskSpeedup = retTask->getSpeedup(ExecEngineConstants::GPU);
			printf("Balancing on\n");
			if( this->gpuThreads*taskSpeedup > taskQueueSize){
				retTask = NULL;
			}else{
#endif
				tasksQueue.erase(it);
#ifdef	LOAD_BALANCING
			}
#endif
		}
	}
	pthread_mutex_unlock(&queueLock);
	return retTask;
}

int TasksQueuePriority::getSize()
{
	int number_tasks = 0;
	pthread_mutex_lock(&queueLock);

	number_tasks = tasksQueue.size();

	pthread_mutex_unlock(&queueLock);

	return number_tasks;
}

/*Task *TasksQueuePriority::getTask(int procType)
{
	Task *retTask = NULL;
	sem_wait(&tasksToBeProcessed);
	pthread_mutex_lock(&queueLock);

	if(tasksQueue.size() > 0){
		multimap<float, Task*>::iterator it;

		if(procType == ExecEngineConstants::GPU){
			it = tasksQueue.end();
			it--;
			retTask = (*it).second;
			tasksQueue.erase(it);
		}else{
			it = tasksQueue.begin();
			retTask = (*it).second;
			tasksQueue.erase(it);
		}
	}
	pthread_mutex_unlock(&queueLock);
	return retTask;
}*/

void TasksQueue::releaseThreads(int numThreads)
{
	// Increment the number of tasks to be processed according to the
	// number of threads accessing this queue. So, all them will get
	// a NULL task, what is interpreted as an end of work.
	for(int i = 0; i < numThreads; i++){
		sem_post(&tasksToBeProcessed);
	}
}





