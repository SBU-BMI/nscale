/*
 * PullHandler_test.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: tcpan
 */




#include "PushCommHandler.h"
#include "Assign.h"
#include "Save.h"
#include "mpi.h"
#include <vector>
#include "Debug.h"
#include "RandomScheduler.h"
#include "MPISendDataBuffer.h"
#include "MPIRecvDataBuffer.h"


int main (int argc, char **argv){
	int threading_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);
//	MPI_Init(&argc, &argv);

	MPI_Comm comm_world;
//	comm_world = MPI_COMM_NULL;
	comm_world = MPI_COMM_WORLD;

	int size, rank;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);

	if (size < 2) {
		cci::common::Debug::print("ERROR:  this test program needs to be run with at least 2 processes.\n");
		MPI_Finalize();
		return -1;
	}

	std::vector<cci::rt::Communicator_I *> handlers;


	cci::rt::Scheduler_I *sch = NULL;
	cci::rt::MPISendDataBuffer *sbuf = NULL;
	cci::rt::MPIRecvDataBuffer *rbuf = NULL;
	cci::rt::CommHandler_I *handler = NULL;

	if (rank % 4 == 0) {
		rbuf = new cci::rt::MPIRecvDataBuffer(100);
		sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
		handler = new cci::rt::PushCommHandler(&comm_world, 0, rbuf, sch, NULL);
	}
	else {
		sbuf = new cci::rt::MPISendDataBuffer(30);
		sch = new cci::rt::RandomScheduler(false, true);
		handler = new cci::rt::PushCommHandler(&comm_world, 0, sbuf, sch, NULL);
	}

	handlers.push_back(handler);
	if (!handler->isListener()) {
		cci::rt::Assign *assign = new cci::rt::Assign(&comm_world, MPI_UNDEFINED, NULL, sbuf, NULL);
		handlers.push_back(assign);
	} else {
		cci::rt::Save *save = new cci::rt::Save(&comm_world, MPI_UNDEFINED, rbuf, NULL, NULL);
		handlers.push_back(save);
	}

	int j = 0;

	int result;
	while (!handlers.empty() ) {
		++j;
		for (std::vector<cci::rt::Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			result = (*iter)->run();
			if (result == cci::rt::Communicator_I::DONE || result == cci::rt::Communicator_I::ERROR) {
				cci::common::Debug::print("%s no output at iter j %d .  DONE or error state %d\n", (*iter)->getClassName(), j, result);
				delete (*iter);
				iter = handlers.erase(iter);
			} else if (result == cci::rt::Communicator_I::READY ) {
				cci::common::Debug::print("%s output generated at iteration j %d.  result = %d\n", (*iter)->getClassName(), j, result);
				++iter;
			} else {
				cci::common::Debug::print("%s no output at iter j %d .  wait state %d\n", (*iter)->getClassName(), j, result);
				++iter;
			}
		}

	}


	MPI_Finalize();

}



