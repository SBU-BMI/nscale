/*
 * Process.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"
#include "PullCommHandler.h"
#include "PushCommHandler.h"
#include "Worker_I.h"
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

	///// first set up the comm handlers

	// for fun, let's set up a compute group and an io group.
	// partitioning is arbitrary.  let the computesize be 3/4 of whole thing.
	// io be 1/4 of whole thing

	CommHandler_I *handler, *handler2;
	std::vector<int> roots;
	std::ostream_iterator<int> out(std::cout, ",");


	// first split into 2.  focus on compute group.
	int g1 = (rank % 4 == 0 ? 1 : 0);  // IO nodes have g1 = 1; compute nodes g1 = 0
	roots.clear();
	roots.push_back(0);

	// compute and io groups
	handler = new PullCommHandler(&comm_world, g1, roots);
	comms.push_back(handler);
	std::cout << "rank " << rank << ": ";
	copy(roots.begin(), roots.end(), out);
	std::cout << std::endl;

	// then the compute to IO communication group
	roots.clear();
	for (int i = 0; i < size; ++i) {
		if (i % 4 == 0) roots.push_back(i);
	}
	handler2 = new PushCommHandler(&comm_world, 0, roots);
	comms.push_back(handler2);
	std::cout << "rank " << rank << ": ";
	copy(roots.begin(), roots.end(), out);
	std::cout << std::endl;

	Debug::print("group 1: %d\n", g1);
	// now set up the workers
	if (g1 == 0) {

		Debug::print("in compute setup\n");
		if (handler->isListener()) {
			Worker_I *assign = new Assign(&comm_world, -1);
			workers.push_back(assign);

			Activity *assignAct = new Activity(NULL, handler, assign);
			activities.push_back(assignAct);
			assignAct->register_listener(listeners);
		} else {
			Worker_I *seg = new Segment(&comm_world, -1);
			workers.push_back(seg);

			Activity *segAct = new Activity(handler, handler2, seg);
			activities.push_back(segAct);
			segAct->register_listener(listeners);
		}

	} else	{


		// then within IO group, split to subgroups, for adios.
		int group_size = 12;
		int group_interleave = 4;
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(*(handler->getComm()), &comm1_size);
		MPI_Comm_rank(*(handler->getComm()), &comm1_rank);

		int g2 = -1;
		int io_root = 0;
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
		// io subgroups
		Worker_I *save = new Save(handler->getComm(), g2);
		workers.push_back(save);
	//		printf("ranks = %d, g1 = %d, Comm1 = %u, g2 = %d, Comm2 = %u\n", rank, g1, *(handler->getComm()), g2, *(handler2->getComm()));
	//	} else {
	//		printf("ranks = %d, g1 = %d, Comm1 = %u \n", rank, g1, *(handler->getComm()));

		Activity *saveAct = new Activity(handler2, NULL, save);
		activities.push_back(saveAct);
		saveAct->register_listener(listeners);
	}

	for (std::tr1::unordered_map<MPI_Comm *, Activity *>::iterator iter = listeners.begin();
			iter != listeners.end(); ++iter) {
		Debug::print("comm is %d activity handler is %lu\n", *(iter->first), iter->second);
	}
//	for (std::tr1::unordered_map<MPI_Comm *, Activity *>::iterator iter = requesters.begin();
//			iter != requesters.end(); ++iter) {
//		Debug::print("comm is %d activity handler is %lu\n", *(iter->first), iter->second);
//	}
	configured = true;

	MPI_Barrier(comm_world);
//	int v = rank;
//	int w;
//	MPI_Allreduce(&v, &w, 1, MPI_INT, MPI_SUM, *(handler->getComm()));
//	printf("ranks = %d, g1 = %d, val = %d\n", rank, g1, w);
}

/**
 * use of deque is potentially a sticky point for synchronization purposes.
 */
void Process::run() {
	Activity * act = NULL;

	Debug::print("listener has %d entries, requester has %d entries\n", listeners.size());
	
	unsigned long working = std::numeric_limits<unsigned long>::max();
	working = working >> (sizeof(unsigned long) * 8 - listeners.size());
	Debug::print("working bit field = %lu\n", working);

	while (!listeners.empty() ) {
		for (std::tr1::unordered_map<MPI_Comm *, Activity *>::iterator iter = listeners.begin();
				iter != listeners.end(); ) {

			if (iter->second->process() == -1) {
				iter = listeners.erase(iter);
			} else ++iter;
		}
//		for (std::tr1::unordered_map<MPI_Comm *, Activity *>::iterator iter = requesters.begin();
//				iter != requesters.end(); ) {
//
//			if (iter->second->process() == -1) {
//				iter = requesters.erase(iter);
//			} else ++iter;
//		}
	}


}


void Process::teardown() {
	MPI_Barrier(comm_world);
	if (!configured) return;

	listeners.clear();
	//requesters.clear();

	// clean up all the communicators.
	for (int i = 0; i < comms.size(); ++i) {
		delete comms[i];
	}
	comms.clear();
	// clean up all the communicators.
	for (int i = 0; i < workers.size(); ++i) {
		delete workers[i];
	}
	workers.clear();
	// clean up all the communicators.
	for (int i = 0; i < activities.size(); ++i) {
		delete activities[i];
	}
	activities.clear();

	configured = false;
}

} /* namespace rt */
} /* namespace cci */
