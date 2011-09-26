/*
 * testQueue.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */


#include "TasksQueue.h"
#include <stdio.h>

int main(){

	printf( "FCFS QUEUE\n" );
	TasksQueueFCFS tasksQueue;

	Task *auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 2.0);
	tasksQueue.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 5.0);
	tasksQueue.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 4.0);
	tasksQueue.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 3.0);
	tasksQueue.insertTask(auxTask);
	tasksQueue.releaseThreads(1);

	auxTask=NULL;
	do{

		auxTask = tasksQueue.getTask(Constant::CPU);
		if(auxTask != NULL){
			printf("Task speedup = %f\n", auxTask->getSpeedup());
			delete auxTask;
		}
	}while(auxTask!= NULL);

	printf("PRIORITY QUEUE\n");

	TasksQueuePriority tasksQueueP;

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 2.0);
	tasksQueueP.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 5.0);
	tasksQueueP.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 4.0);
	tasksQueueP.insertTask(auxTask);

	auxTask = new Task();
	auxTask->setSpeedup(Constant::GPU, 3.0);
	tasksQueueP.insertTask(auxTask);
	tasksQueueP.releaseThreads(1);

	auxTask=NULL;
	do{

		auxTask = tasksQueueP.getTask(Constant::CPU);
		if(auxTask != NULL){
			printf("Task speedup = %f\n", auxTask->getSpeedup());
			delete auxTask;
		}
	}while(auxTask!= NULL);



	return 0;
}
