/*
 * Process.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"
#include "PullCommHandler.h"
#include "PushCommHandler.h"

#include <iostream>
#include <string.h>

namespace cci {
namespace rt {

Process::Process(int argc, char **argv) : configured(false) {
	// common initialization
	int threading_provided;
	int err  = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	gethostname(hostname, 255);  // from <iostream>

	comm_world = MPI_COMM_WORLD;

}

Process::~Process() {
	teardown();

	MPI_Finalize();
}


void Process::setup() {
	// if already configured, clean up,
	teardown();

	int size, rank;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);

	// now do a set up.

	// for fun, let's set up a compute group and an io group.
	// partitioning is arbitrary.  let the computesize be 3/4 of whole thing.
	// io be 1/4 of whole thing

	MPI_Comm comm;
	CommHandler_I *handler, *handler2;
	std::vector<int> roots;

	// first split into 2.
	int g1 = (rank % 4 == 0 ? 1 : 0);  // IO nodes have g1 = 0; compute nodes g1 = 1
	roots.clear();
	roots.push_back(0);

	handler = new PullCommHandler(comm_world, g1, roots);

	handlers.push_back(handler);


	// then within IO group, do subgroups.
	int group_size = 12;
	int group_interleave = 4;
	int comm1_size;
	int comm1_rank;
	MPI_Comm_size(handler->getComm(), &comm1_size);
	MPI_Comm_rank(handler->getComm(), &comm1_rank);

	int g2 = -1;
	int io_root = 0;
	if (g1 == 1) {
		if (group_size == 1) {
			g2 = comm1_rank;
		} else if (group_size < 1) {
			if (comm1_rank == io_root) g2 = 0;
			else g2 = 1;
		} else {
			if (comm1_rank == io_root) g2 = 0;
			else {
				if (group_interleave > 1) {
					int blockid = comm1_rank / (group_size * group_interleave);
					g2 = blockid * group_interleave + comm1_rank % group_interleave;
				} else {
					g2 = comm1_rank / group_size;
				}
				++g2;
			}
		}
		handler2 = new PushCommHandler(handler->getComm(), g2, roots)
		handlers.push_back(handler2);
	}



	printf("ranks = %d, g1 = %d, Comm1 = %u, g2 = %d, Comm2 = %u\n", rank, g1, handler->getComm(), g2, handler2->getComm());

	int v = rank;
	int w;
	MPI_Allreduce(&v, &w, 1, MPI_INT, MPI_SUM, handler->getComm());
	printf("ranks = %d, g1 = %d, val = %d\n", rank, g1, w);
}

/**
 * use of deque is potentially a sticky point for synchronization purposes.
 */
void Process::run() {
//	CommHandler * h = NULL;
//	while (!handlers.empty()) {
//		h = handlers.front();
//		handlers.pop_front();
//
//		if (h->process() != CommHandler::FINISHED)
//			handlers.push_back(h);
//
//	}

}


void Process::teardown() {
	MPI_Barrier(comm_world);
	if (!configured) return;
	// clean up all the communicators.


//	for (int i = 0; i < 2; i++) {
//		if (comms[i] != MPI_COMM_NULL) MPI_Comm_free(comms + i);
//		else printf("null comm \n");
//	}
//	delete [] comms;

//	for (int i = comms.size() - 1; i >= 0; --i) {
//		if (comms[i] != MPI_COMM_NULL) MPI_Comm_free(&comms[i]);
//	}

	for (int i = 0; i < handlers.size(); ++i) {
		delete handlers[i];
	}
	handlers.clear();
	configured = false;
}

} /* namespace rt */
} /* namespace cci */
