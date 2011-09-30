/*
 * ThreadPool.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#include "ThreadPool.h"
#include <sched.h>
#include <string.h>


// GPU functions called to initialize device.
void warmUp(int device);

void *callThread(void *arg){
	ThreadPool *tp = (ThreadPool *)((threadData*) arg)->execEnginePtr;
	int procType = (int)((threadData*) arg)->procType;
	int tid = (int)((threadData*) arg)->tid;

	// If threads is managing GPU, than init adequate device
	if(procType == 2){
		printf("WarnUP: GPU id = %d\n", tid);
		warmUp(tid);
		int cpuId=tid;

#ifdef	LINUX
		if(tid==2){
			cpuId = 11;
		}
		cpu_set_t cpu_info;
		CPU_ZERO(&cpu_info);
		CPU_SET(cpuId, &cpu_info);

		if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_info) == -1) {
			printf("Error: sched_getaffinity\n");
		}
#endif
	}

	tp->processTasks(procType, tid);
	free(arg);
	pthread_exit(NULL);
}

ThreadPool::ThreadPool(TasksQueue *tasksQueue) {
	// init thread pool with user provided task queue
	this->tasksQueue = tasksQueue;

	CPUWorkerThreads = NULL;
	GPUWorkerThreads = NULL;

	pthread_mutex_init(&initExecutionMutex, NULL);
	pthread_mutex_lock(&initExecutionMutex);

	pthread_mutex_init(&createdThreads, NULL);
	pthread_mutex_lock(&createdThreads);

	numGPUThreads = 0;
	numCPUThreads = 0;

	firstToFinish=true;
}

ThreadPool::~ThreadPool() {
	this->finishExecWaitEnd();

	if(CPUWorkerThreads != NULL){
		free(CPUWorkerThreads);
	}

	if(GPUWorkerThreads != NULL){
		free(GPUWorkerThreads);
	}
	pthread_mutex_destroy(&initExecutionMutex);

	float loadImbalance = 0.0;
	if(numCPUThreads+numGPUThreads > 1){
		// calculate time in microseconds
		double tS = firstToFinishTime.tv_sec*1000000 + (firstToFinishTime.tv_usec);
		double tE = lastToFinishTime.tv_sec*1000000  + (lastToFinishTime.tv_usec);
		loadImbalance = (tE - tS)/1000000.0;
	}
	printf("Load imbalance = %f\n", loadImbalance);

}

bool ThreadPool::createThreadPool(int cpuThreads, int *cpuThreadsCoreMapping, int gpuThreads, int *gpuThreadsCoreMapping)
{
	//TODO: assign threads to processors.....
	// Initialize mutexes used to control the status of the worker threads

	// Create CPU threads.
	if(cpuThreads > 0){
		numCPUThreads = cpuThreads;
		CPUWorkerThreads = (pthread_t *) malloc(sizeof(pthread_t) * cpuThreads);

		for (int i = 0; i < cpuThreads; i++ ){
			threadData *arg = (threadData *) malloc(sizeof(threadData));
			arg->tid = i;
			arg->procType = Constant::CPU;
			arg->execEnginePtr = this;
			int ret = pthread_create(&(CPUWorkerThreads[arg->tid]), NULL, callThread, (void *)arg);
			if (ret){
				printf("ERROR: Return code from pthread_create() is %d\n", ret);
				exit(-1);
			}

			// wait untill thead is created
			pthread_mutex_lock(&createdThreads);
		}
	}
	// Create CPU threads.
	if(gpuThreads > 0){
		numGPUThreads = gpuThreads;
		GPUWorkerThreads = (pthread_t *) malloc(sizeof(pthread_t) * gpuThreads);

		for (int i = 0; i < gpuThreads; i++ ){
			threadData *arg = (threadData *) malloc(sizeof(threadData));
			arg->tid = i;
			arg->procType = Constant::GPU;
			arg->execEnginePtr = this;
			int ret = pthread_create(&(GPUWorkerThreads[arg->tid]), NULL, callThread, (void *)arg);
			if (ret){
				printf("ERROR: Return code from pthread_create() is %d\n", ret);
				exit(-1);
			}
			// wait untill thead is created
			pthread_mutex_lock(&createdThreads);

		}
	}
	for(int i = 0; i < cpuThreads + gpuThreads; i++){
		curProcTasks.push_back(NULL);
	}

	return true;
}


void ThreadPool::initExecution()
{
	pthread_mutex_unlock(&initExecutionMutex);
}

void sighand(int signo)
{
	printf("Thread got signal. Leaving\n");
	pthread_exit(NULL); 
}

void ThreadPool::taskReplicationAction(int procType){
	struct sigaction        actions;

	printf("Enter Testcase\n" );

	printf("Set up the alarm handler for the process\n");
	memset(&actions, 0, sizeof(actions));
	sigemptyset(&actions.sa_mask);
	actions.sa_flags = 0;
	actions.sa_handler = sighand;

	for(int i=0; i<this->numCPUThreads; ++i) {
		// this threads still processing.
		if(curProcTasks[this->numGPUThreads+i] !=NULL){
			// kill CPU thread and process the task
 			int rc = pthread_kill(CPUWorkerThreads[i], SIGALRM);
			printf("Kill CPU thread id = %d\n", i);
		}
	}

}

void ThreadPool::processTasks(int procType, int tid)
{
	// Inform that this threads was created.
	//sem_post(&createdThreads);
	pthread_mutex_unlock(&createdThreads);

	printf("procType:%d  tid:%d waiting init of execution\n", procType, tid);
	pthread_mutex_lock(&initExecutionMutex);
	pthread_mutex_unlock(&initExecutionMutex);

	Task *curTask = NULL;
	
	// Id used by the thread to access the array of tasks being processed
	int threadGlobalId = tid;

	if(procType == Constant::CPU){
		threadGlobalId += this->numGPUThreads; 
	}

	printf("Begin: procType:%d  tid:%d globalId=%d\n", procType, tid, threadGlobalId);

	while(true){

		curTask = this->tasksQueue->getTask(procType);
		if(curTask == NULL){
			printf("procType:%d  tid:%d Task NULL\n", procType, tid);
#ifdef	TASK_REPLICATION
			if(procType == Constant::GPU){
				taskReplicationAction(procType);
			}
			sleep(10);
#endif
			break;
		}
		
		curProcTasks[threadGlobalId] = curTask;

		curTask->run(procType);

		curProcTasks[threadGlobalId] = NULL;

		printf("procType:%d  tid:%d Task speedup = %f\n", procType, tid, curTask->getSpeedup());
		delete curTask;

	}

	printf("Leaving procType:%d  tid:%d\n", procType, tid);

	// I don't care about controling concurrent access here. Whether two threads 
	// enter this if, their resulting gettimeofday will be very similar if not the same.
	if(firstToFinish){
		firstToFinish = false;
		gettimeofday(&firstToFinishTime, NULL);
	}

	gettimeofday(&lastToFinishTime, NULL);
}

int ThreadPool::getGPUThreads()
{
	return numGPUThreads;
}



void ThreadPool::finishExecWaitEnd()
{
	// make sure to init the execution whether if was not done by the user.
	initExecution();

	for(int i= 0; i < numCPUThreads; i++){
		pthread_join(CPUWorkerThreads[i] , NULL);
	}
	for(int i= 0; i < numGPUThreads; i++){
		pthread_join(GPUWorkerThreads[i] , NULL);
	}

}

int ThreadPool::getCPUThreads()
{
	return numCPUThreads;
}





