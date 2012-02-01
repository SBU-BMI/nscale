

#include <stdio.h>
#include <sys/time.h>
#include "TaskSleep.h"
#include "ExecutionEngine.h"

#define NUM_TASKS	10

int main(int argc, char **argv){
	
	ExecutionEngine execEngine(1, 1, 1);

	for(int i =0; i < NUM_TASKS; i++){
		TaskSleep *ts = new TaskSleep();
		ts->setSpeedup(ExecEngineConstants::GPU, 2.0);
		execEngine.insertTask(ts);

	}

	execEngine.startupExecution();
	execEngine.endExecution();
	
	return 0;
}


