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
#include <sstream>

#include "Debug.h"


#include <string.h>
#include <iterator>
#include <limits>

namespace cci {
namespace rt {

Process::Process(MPI_Comm &_comm_world, int argc, char **argv, ProcessConfigurator_I *_conf) :
		conf(_conf), configured(false), comm_world(_comm_world) {
	// common initialization
	handlers.clear();

	world_rank = MPI_UNDEFINED;
	MPI_Comm_rank(comm_world, &world_rank);

	memset(hostname, 0, 256);

	if (conf != NULL) conf->init();

}

Process::~Process() {
	//Debug::print("Process destructor called\n");

	teardown();

	if (conf != NULL) conf->finalize();



}


void Process::setup() {
	// if already configured, clean up,
	if (configured) teardown();

	configured = conf->configure(comm_world, this);


	MPI_Barrier(comm_world);

	if (world_rank == 0) Debug::print("Processes configured\n");
}

/**
 * use of deque is potentially a sticky point for synchronization purposes.
 */
void Process::run() {

	if (!configured) {
		Debug::print("ERROR:  not configured\n");
		return;
	}
	//Debug::print("Process running\n");

//	unsigned long working = std::numeric_limits<unsigned long>::max();
//	working = working >> (sizeof(unsigned long) * 8 - handlers.size());
//	Debug::print("listener has %d entries, working bit field = %x\n", handlers.size(), working);

	std::stringstream ss;

	int result;
	while (!handlers.empty() ) {

		for (std::vector<Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			result = (*iter)->run();
			if (result == Communicator_I::DONE || result == Communicator_I::ERROR) {
				//ss.str(std::string());
				//ss << "DELETING " << (*iter)->getClassName() << ". ";
//				Communicator_I::dereference((*iter), &handlers);
				delete (*iter);
				iter = handlers.erase(iter);

//				if (!handlers.empty()) {
//
//					ss << "Handlers remaining: ";
//					for (std::vector<Communicator_I *>::iterator iter2 = handlers.begin();
//							iter2 != handlers.end(); ++iter2) {
//						ss << "(" << (*iter2)->getClassName() << ":" << (*iter2)->getStatus() << "), ";
//					}
//				}
				//Debug::print("Process %s\n", ss.str().c_str());
			} else ++iter;
		}
	}

}


void Process::teardown() {
	MPI_Barrier(comm_world);

	if (!configured) return;

	// clean up all the communication handlers.
	for (int i = 0; i < handlers.size(); ++i) {
//		Communicator_I::dereference(handlers[i], &handlers);
		delete handlers[i];
	}
	handlers.clear();

	configured = false;
}

} /* namespace rt */
} /* namespace cci */
