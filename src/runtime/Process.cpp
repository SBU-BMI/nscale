/*
 * Process.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"
#include "PullCommHandler.h"
#include "PushCommHandler.h"
#include "Action_I.h"

#include "Debug.h"


#include <iostream>
#include <string.h>
#include <iterator>
#include <limits>

namespace cci {
namespace rt {

Process::Process(int argc, char **argv, ProcessConfigurator_I *_conf) :
		conf(_conf), configured(false) {
	// common initialization
	int threading_provided;
	int err  = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	gethostname(hostname, 255);  // from <iostream>

	comm_world = MPI_COMM_WORLD;
	MPI_Comm_rank(comm_world, &world_rank);

	logger = new cciutils::SCIOLogger(world_rank, hostname, 0);

	if (conf != NULL) conf->init(logger);

}

Process::~Process() {
	//Debug::print("Process destructor called\n");

	if (configured) teardown();

	if (conf != NULL) {
		conf->finalize();
		delete conf;
	}

#if defined (WITH_MPI)
	logger->writeCollectively("test.log", world_rank, 0, comm_world);
#else
	logger->write("test.log");
#endif

	if (logger) delete logger;

	MPI_Finalize();
}


void Process::setup() {
	// if already configured, clean up,
	if (configured) teardown();

	configured = conf->configure(comm_world, this);

	Debug::print("Process configured\n");

	MPI_Barrier(comm_world);
}

/**
 * use of deque is potentially a sticky point for synchronization purposes.
 */
void Process::run() {

	if (!configured) {
		Debug::print("ERROR:  not configured\n");
		return;
	}
	Debug::print("Process running\n");

//	unsigned long working = std::numeric_limits<unsigned long>::max();
//	working = working >> (sizeof(unsigned long) * 8 - handlers.size());
//	Debug::print("listener has %d entries, working bit field = %x\n", handlers.size(), working);

	int result;
	while (!handlers.empty() ) {
		for (std::vector<Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			result = (*iter)->run();
			if (result == Communicator_I::DONE || result == Communicator_I::ERROR) {
				Communicator_I::dereference((*iter), &handlers);
				iter = handlers.erase(iter);
			} else ++iter;
		}
	}

}


void Process::teardown() {
	MPI_Barrier(comm_world);

	if (!configured) return;

	// clean up all the communication handlers.
	for (int i = 0; i < handlers.size(); ++i) {
		Communicator_I::dereference(handlers[i], &handlers);
	}
	handlers.clear();

	configured = false;
}

} /* namespace rt */
} /* namespace cci */
