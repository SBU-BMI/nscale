/*
 * Process.h
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#ifndef PROCESS_H_
#define PROCESS_H_

#include <vector>
#include "Activity_I.h"
#include "mpi.h"

namespace cci {
namespace rt {



class Process {
public:
	Process(int argc, char **argv);
	virtual ~Process();

	/**
	 * Initializes and configures MPI processes: connectivity graph in a comm, comm splits
	 *
	 * implementation of this function is meant to be offloaded somewhere else
	 * or generalized such as reading from an xml file and then generating the layout
	 *
	 * this is also where the mapping of the handlers are done.
	 */
	virtual void setup();
	/**
	 * execution loop, iterate over handlers
	 */
	virtual void run();
	/**
	 * clean up.  free communicators.
	 */
	virtual void teardown();

private:
	/**
	 * queue of handlers.  double ended because response time for the handler may be different.
	 */
	std::vector<Activity_I *> activities;
	std::vector<CommHandler_I *> handlers;
	MPI_Comm *comms;

	MPI_Comm comm_world;
	char hostname[256];
	bool configured;
};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESS_H_ */
