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
#include "Assign.h"
#include "Save.h"
#include "Segment.h"
#include "Debug.h"

#include <iostream>
#include <string.h>
#include <iterator>
#include <limits>
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
	int g1=-1, g2=-1, g3 = -1;


	///// first set up the comm handlers

	// for fun, let's set up a compute group and an io group.
	// partitioning is arbitrary.  let the computesize be 3/4 of whole thing.
	// io be 1/4 of whole thing

	RootedCommHandler_I *handler, *handler2;
	std::vector<int> roots;
//	std::ostream_iterator<int> out(std::cout, ",");


	// first split into 2.  focus on compute group.
	g1 = (rank % 4 == 0 ? 1 : 0);  // IO nodes have g1 = 1; compute nodes g1 = 0
	roots.clear();
	roots.push_back(1);

	// compute and io groups
	handler = new PullCommHandler(&comm_world, g1, roots);

	// then the compute to IO communication group
	g3 = (g1 == 0 && handler->isListener() ? 2: 3);
	roots.clear();
	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0) roots.push_back(i);
	}
	handler2 = new PushCommHandler(&comm_world, g3, roots);
	//std::cout << "rank " << rank << ": ";
	//copy(roots.begin(), roots.end(), out);
	//std::cout << std::endl;

	// now set up the workers
	if (g1 == 0) {
		handlers.push_back(handler);
		//Debug::print("in compute setup\n");
		if (g3 == 2) {
			Action_I *assign = new Assign(&comm_world, -1);
			handlers.push_back(assign);
			handler->setAction(assign);
			delete handler2;
		} else {
			Action_I *seg = new Segment(&comm_world, -1);
			handlers.push_back(seg);
			handlers.push_back(handler2);
			handler->setAction(seg);
			handler2->setAction(seg);
		}

	} else	{


		// then within IO group, split to subgroups, for adios.
		int group_size = 12;
		int group_interleave = 4;
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(*(handler->getComm()), &comm1_size);
		MPI_Comm_rank(*(handler->getComm()), &comm1_rank);

		int io_root = 0;
		if (group_size == 1) {
			g2 = comm1_rank;
		} else if (group_size < 1) {
			if (comm1_rank == io_root) g2 = 3;
			else g2 = 4;
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
		// io subgroups
		Action_I *save = new Save(handler->getComm(), g2);  // comm is group 1 IO comms, split into g2 comms
		handlers.push_back(handler2);
		handlers.push_back(save);
		handler2->setAction(save);
		delete handler;
	}

	configured = true;



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
