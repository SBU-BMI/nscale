/*
 * Process.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#ifndef PROCESS_H_
#define PROCESS_H_

#include "mpi.h"
#include <vector>
#include <tr1/unordered_map>
#include "Schedule.h"
#include "Work.h"

namespace cci {

namespace runtime {

/**
 * represents a set of processes.
 *
 * to initialize, first create a manager process, then create layers of managers,
 * then create the workers.
 *
 * at the same time or after the processes are created, assign work objects
 *
 * finally call connect to connect sets of src and dest as fully connected graph
 *
 * purpose of this is to manage the MPI node setup, and also manage the control messages.
 */
class Process {
public:
	Process(MPI_Comm *global_comm, int global_rank,
			MPI_Comm *local_comm, int local_rank, Process *_manager, const int queueSize);
	virtual ~Process();

	/**
	 * this method connects a set of output to a set of input, and associate schedules
	 * LATER:  we can disconnect and reconnect.,
	 */
	static int connect(std::vector<Process *> &_src, std::vector<Process *> &_dest,
			Schedule *srcSchedule, Schedule *destSchedule);
	static int disconnectAll() {};

	static int setWork(std::vector<Process *> &_nodes, std::vector<Work *> &_workTypes);

	/**
	 * runs and messages between nodes.  called from main.
	 */
	virtual int execute();

	virtual int getStatus() { return status; };

private:
	// local comm for the group of processes
	MPI_Comm *local_comm;
	int local_rank;

	// global comm for everyone
	MPI_Comm *global_comm;
	int global_rank;

	// each process has one or more input, and one or more output
	std::vector<std::vector<Process *> > inputProcs;
	std::vector<std::vector<Process *> > outputProcs;
	// input and output are associated with some strategy.
	std::vector<Schedule *> parentStrategies;
	std::vector<Schedule *> childStrategies;

	// and a management process
	Process *managerProc;

	// each process also has a set of work associated with it.
	// at each point in time, only one work item can run.
	std::tr1::unordered_map<int, Work *> workTypes;

	// each process also has a queue of parameters.  parameter indicates work type
	ParamQueue queues;

	int status;
};

}

}

#endif /* PROCESS_H_ */
