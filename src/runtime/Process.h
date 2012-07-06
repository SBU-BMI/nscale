/*
 * Process.h
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#ifndef PROCESS_H_
#define PROCESS_H_

#include <vector>
#include "Communicator_I.h"
#include "ProcessConfigurator_I.h"

#include "mpi.h"

namespace cci {
namespace rt {

class ProcessConfigurator_I;

class Process {
public:
	Process(int argc, char **argv, ProcessConfigurator_I *_conf);
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

	void addHandler(Communicator_I * handler) {
		handlers.push_back(handler);
		Communicator_I::reference(handler, &handlers);
	}

private:
	/**
	 * queue of handlers.  double ended because response time for the handler may be different.
	 */
	std::vector<Communicator_I *> handlers;

	ProcessConfigurator_I *conf;

	MPI_Comm comm_world;
	char hostname[256];
	bool configured;
};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESS_H_ */
