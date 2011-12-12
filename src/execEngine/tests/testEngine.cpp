/*
 * testQueue.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#include "ExecutionEngine.h"

int main(){
	ExecutionEngine execEngine(1, 1, 2);

	for(int i = 0; i < 20; i++){
		Task *auxTask = new Task();
		auxTask->setSpeedup(ExecEngineConstants::GPU, 1.0);
		execEngine.insertTask(auxTask);
		auxTask = new Task();
		auxTask->setSpeedup(ExecEngineConstants::GPU, 20.0);
		execEngine.insertTask(auxTask);
	}

	execEngine.startupExecution();
	execEngine.endExecution();

	return 0;
}
