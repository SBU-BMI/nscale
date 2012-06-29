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

}

Process::~Process() {
	teardown();

	if (conf != NULL) delete conf;

	MPI_Finalize();
}


void Process::setup() {
	// if already configured, clean up,
	teardown();

	configured = conf->configure(comm_world, handlers);

	MPI_Barrier(comm_world);
}

/**
 * use of deque is potentially a sticky point for synchronization purposes.
 */
void Process::run() {

	unsigned long working = std::numeric_limits<unsigned long>::max();
	working = working >> (sizeof(unsigned long) * 8 - handlers.size());
	Debug::print("listener has %d entries, working bit field = %x\n", handlers.size(), working);

	int result;
	while (!handlers.empty() ) {
		for (std::vector<Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			result = (*iter)->run();
			if (result == Communicator_I::DONE || result == Communicator_I::ERROR) {
				if ((*iter)->dereference(&handlers) == 0) {
					delete (*iter);
				}
				iter = handlers.erase(iter);
			} else ++iter;
		}
	}

}


void Process::teardown() {
	MPI_Barrier(comm_world);
	if (!configured) return;

	// clean up all the communicators.
	for (int i = 0; i < handlers.size(); ++i) {
		if (handlers[i]->dereference(&handlers) == 0) delete handlers[i];
	}
	handlers.clear();

	configured = false;
}

} /* namespace rt */
} /* namespace cci */
