/*
 * SegConfigurator.cpp
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#include "SegConfigurator.h"

#include "PullCommHandler.h"
#include "PushCommHandler.h"
#include "Save.h"
#include "Assign.h"
#include "Segment.h"
#include "RandomScheduler.h"


namespace cci {
namespace rt {

SegConfigurator::SegConfigurator() {
	// TODO Auto-generated constructor stub

}

SegConfigurator::~SegConfigurator() {
	// TODO Auto-generated destructor stub
}

bool SegConfigurator::configure(MPI_Comm &comm, std::vector<Communicator_I *> &handlers) {

	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// now do a set up.
	int g1=-1, g2=-1, g3 = -1;


	///// first set up the comm handlers

	// for fun, let's set up a compute group and an io group.
	// partitioning is arbitrary.  let the computesize be 3/4 of whole thing.
	// io be 1/4 of whole thing

	CommHandler_I *handler, *handler2;
	std::vector<int> roots;
//	std::ostream_iterator<int> out(std::cout, ",");


	// first split into 2.  focus on compute group.
	g1 = (rank % 4 == 0 ? 1 : 0);  // IO nodes have g1 = 1; compute nodes g1 = 0
	Scheduler_I *sch = NULL;
	if (rank == 1) sch = new RandomScheduler(true, false);  // root at rank = 0
	else sch = new RandomScheduler(false, true);

	// compute and io groups
	handler = new PullCommHandler(&comm, g1, sch);

	// then the compute to IO communication group
	g3 = (g1 == 0 && handler->isListener() ? 2: 3);

	Scheduler_I *sch2 = NULL;
	if (rank % 4 == 0) sch2 = new RandomScheduler(true, false);  // root at rank = 0
	else sch2 = new RandomScheduler(false, true);

	handler2 = new PushCommHandler(&comm, g3, sch2);
	//std::cout << "rank " << rank << ": ";
	//copy(roots.begin(), roots.end(), out);
	//std::cout << std::endl;

	// now set up the workers
	if (g1 == 0) {
		handlers.push_back(handler);
		//Debug::print("in compute setup\n");
		if (g3 == 2) {
			Action_I *assign = new Assign(&comm, -1);
			handlers.push_back(assign);
			handler->setAction(assign);
			delete handler2;
		} else {
			Action_I *seg = new Segment(&comm, -1);
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
	return true;
}

} /* namespace rt */
} /* namespace cci */
