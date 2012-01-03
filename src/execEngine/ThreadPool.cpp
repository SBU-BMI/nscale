/*
 * ThreadPool.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#include "ThreadPool.h"
#include <sched.h>


// GPU functions called to initialize device.
//void warmUp(int device);
//void *cudaMemAllocWrapper(int dataSize);
//void cudaFreeMemWrapper(void *data_ptr);

void *callThread(void *arg){
	ThreadPool *tp = (ThreadPool *)((threadData*) arg)->execEnginePtr;
	int procType = (int)((threadData*) arg)->procType;
	int tid = (int)((threadData*) arg)->tid;

	// If threads is managing GPU, than init adequate device
	if(procType == 2){
		printf("WarnUP: GPU id = %d\n", tid);
//		warmUp(tid);

		int cpuId=tid;

////		cpu_set_t cpu_info;
////		CPU_ZERO(&cpu_info);
////		if(tid==2){
////			cpuId=11;
////		}
////		CPU_SET(cpuId, &cpu_info);
////
////
////		if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_info) == -1) {
////			printf("Error: sched_getaffinity\n");
////		}

	}else{
		int cpuId=tid+2;

////		cpu_set_t cpu_info;
////		CPU_ZERO(&cpu_info);
////		CPU_SET(cpuId, &cpu_info);
////
////
////		if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_info) == -1) {
////			printf("Error: sched_getaffinity\n");
////		}



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
	gpuTempDataSize = 0;
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

void * ThreadPool::getGPUTempData(int tid){
	return this->gpuTempData[tid];
}

bool ThreadPool::createThreadPool(int cpuThreads, int *cpuThreadsCoreMapping, int gpuThreads, int *gpuThreadsCoreMapping, int gpuTempDataSize)
{

	this->gpuTempDataSize = gpuTempDataSize;

	// Create CPU threads.
	if(cpuThreads > 0){
		numCPUThreads = cpuThreads;
		CPUWorkerThreads = (pthread_t *) malloc(sizeof(pthread_t) * cpuThreads);

		for (int i = 0; i < cpuThreads; i++ ){
			threadData *arg = (threadData *) malloc(sizeof(threadData));
			arg->tid = i;
			arg->procType = ExecEngineConstants::CPU;
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
			arg->procType = ExecEngineConstants::GPU;
			arg->execEnginePtr = this;
			int ret = pthread_create(&(GPUWorkerThreads[arg->tid]), NULL, callThread, (void *)arg);
			if (ret){
				printf("ERROR: Return code from pthread_create() is %d\n", ret);
				exit(-1);
			}
			gpuTempData.push_back(NULL);
			// wait untill thead is created
			pthread_mutex_lock(&createdThreads);

		}
	}

	return true;
}


void ThreadPool::initExecution()
{
	pthread_mutex_unlock(&initExecutionMutex);
}


void ThreadPool::processTasks(int procType, int tid)
{
	// Inform that this threads was created.
	//sem_post(&createdThreads);
	pthread_mutex_unlock(&createdThreads);

//	if(ExecEngineConstants::GPU == procType && gpuTempDataSize >0){
//		printf("GPU tid=%d allocDataSize=%d\n", tid, gpuTempDataSize);
//		gpuTempData[tid] = cudaMemAllocWrapper(gpuTempDataSize);
//	}

	printf("procType:%d  tid:%d waiting init of execution\n", procType, tid);
	pthread_mutex_lock(&initExecutionMutex);
	pthread_mutex_unlock(&initExecutionMutex);

	Task *curTask = NULL;
// ProcessTime example
struct timeval startTime;
struct timeval endTime;

	while(true){

		curTask = this->tasksQueue->getTask(procType);
		if(curTask == NULL){
			printf("procType:%d  tid:%d Task NULL\n", procType, tid);
			break;
		}
	gettimeofday(&startTime, NULL);

		curTask->run(procType, tid);

		gettimeofday(&endTime, NULL);
		// calculate time in microseconds
		double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
		double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
		printf("procType:%d  tid:%d Task speedup = %f procTime = %f\n", procType, tid, curTask->getSpeedup(), (tE-tS)/1000000);
		delete curTask;

	}

	printf("Leaving procType:%d  tid:%d\n", procType, tid);
//	if(ExecEngineConstants::GPU == procType && gpuTempDataSize >0){
//		cudaFreeMemWrapper(gpuTempData[tid]);
//	}
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





